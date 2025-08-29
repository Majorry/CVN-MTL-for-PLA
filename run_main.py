import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from utils.data_loader import DatasetLoader
from utils.statics import StatUtils
from utils.parser import parse_args
from utils.scheduler import WarmUpCosineAnnealingLR, FakeLR
from utils import logger
from utils.init import init_device, init_model
from datetime import datetime
import optuna


def objective(trial, args, train_loader, test_loader, device):
    # --------------------------
    # 1. 超参数空间定义（核心）
    # --------------------------
    # 用Optuna采样超参数，覆盖原args中的默认值
    hparams = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),  # 学习率（对数尺度采样）
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),  # 批大小
        "reg_weight": trial.suggest_float("reg_weight", 0.1, 0.9),  # 回归损失权重
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True),  # 权重衰减
        "epochs": trial.suggest_int("epochs", 20, 100)  # 训练轮次（可选，也可固定）
    }

    # 更新args中的超参数（用采样值覆盖）
    args.lr = hparams["lr"]
    args.batch_size = hparams["batch_size"]
    reg_weight = hparams["reg_weight"]

    # --------------------------
    # 2. 初始化模型和优化器
    # --------------------------
    model = init_model(args)
    model.to(device)

    criterion_class = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"]
    )

    # 学习率调度器（保持原有逻辑）
    scheduler = (
        FakeLR(optimizer) if args.scheduler == 'const' else
        WarmUpCosineAnnealingLR(
            optimizer,
            T_max=hparams["epochs"] * len(train_loader),
            T_warmup=30 * len(train_loader),
            eta_min=7e-5
        )
    )

    # --------------------------
    # 3. 训练模型（简化版，只保留核心逻辑）
    # --------------------------
    best_f1 = 0.0  # 用F1分数作为优化目标
    for epoch in range(hparams["epochs"]):
        # 训练一轮
        train_one_epoch(
            model, train_loader, criterion_class, criterion_reg,
            optimizer, device, reg_weight=reg_weight
        )
        scheduler.step()

        # 验证性能（用test_model获取F1）
        test_results = test_model(model, test_loader, device)
        current_f1 = test_results["f1"]

        # 剪枝（可选，提前终止差的试验）
        trial.report(current_f1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # 更新最佳F1
        if current_f1 > best_f1:
            best_f1 = current_f1

    return best_f1  # 目标：最大化F1分数





def save_best_model(model, args, test_accuracy, test_mse, best_accuracy, best_mse, model_save_dir):
    if test_accuracy > best_accuracy and test_mse < best_mse:
        best_accuracy = test_accuracy
        best_mse = test_mse
        model_save_path = os.path.join(
            model_save_dir,
            f"MTL_range{args.range}_{args.model}_"
            f"cl{args.user}_bz{args.batch_size}_epo{args.epochs}_"
            f"lr{args.lr}_acc{test_accuracy:.4f}_mse{test_mse:.4f}.pt"
        )

        torch.save({
            'state_dict': model.state_dict(),
            'args': vars(args)
        }, model_save_path)
        print(f"Saved model with test accuracy: {test_accuracy:.4f}% to {model_save_path}")
    return best_accuracy, best_mse


def train_one_epoch(model, train_loader, criterion_class, criterion_reg, optimizer, device, reg_weight=0.5):
    model.train()
    running_total = 0.0
    running_class = 0.0
    running_reg = 0.0
    correct = 0
    total = 0

    for inputs, (auc_labels, loc_labels) in tqdm(train_loader, desc="Training"):
        inputs = inputs.to(device)
        auc_labels = auc_labels.to(device)
        loc_labels = loc_labels.float().to(device)

        optimizer.zero_grad()
        class_out, loc_out = model(inputs)

        # Calculate losses
        loss_class = criterion_class(class_out, auc_labels)
        loss_reg = criterion_reg(loc_out, loc_labels)
        total_loss = (1 - reg_weight) * loss_class + reg_weight * loss_reg

        total_loss.backward()
        optimizer.step()

        # Update statistics
        running_total += total_loss.item() * inputs.size(0)
        running_class += loss_class.item() * inputs.size(0)
        running_reg += loss_reg.item() * inputs.size(0)

        _, predicted = torch.max(class_out, 1)
        total += auc_labels.size(0)
        correct += (predicted == auc_labels).sum().item()

    return {
        'total_loss': running_total / len(train_loader.dataset),
        'class_loss': running_class / len(train_loader.dataset),
        'reg_loss': running_reg / len(train_loader.dataset),
        'accuracy': 100 * correct / total
    }


def evaluate_model(model, test_loader, criterion_class, criterion_reg, device, reg_weight=0.1):
    model.eval()
    running_total = 0.0
    running_class = 0.0
    running_reg = 0.0
    correct = 0
    total = 0
    all_loc_preds = []
    all_loc_labels = []

    with torch.no_grad():
        for inputs, (auc_labels, loc_labels) in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            auc_labels = auc_labels.to(device)
            loc_labels = loc_labels.float().to(device)

            class_out, loc_out = model(inputs)

            # Calculate losses
            loss_class = criterion_class(class_out, auc_labels)
            loss_reg = criterion_reg(loc_out, loc_labels)
            total_loss = (1 - reg_weight) * loss_class + reg_weight * loss_reg

            # Update statistics
            running_total += total_loss.item() * inputs.size(0)
            running_class += loss_class.item() * inputs.size(0)
            running_reg += loss_reg.item() * inputs.size(0)

            # Classification
            _, predicted = torch.max(class_out, 1)
            total += auc_labels.size(0)
            correct += (predicted == auc_labels).sum().item()

            # Regression
            all_loc_preds.append(loc_out.cpu().numpy())
            all_loc_labels.append(loc_labels.cpu().numpy())

    # Calculate regression metrics
    loc_preds = np.concatenate(all_loc_preds)
    loc_labels = np.concatenate(all_loc_labels)
    mse = mean_squared_error(loc_labels, loc_preds)
    mae = mean_absolute_error(loc_labels, loc_preds)

    return {
        'total_loss': running_total / len(test_loader.dataset),
        'class_loss': running_class / len(test_loader.dataset),
        'reg_loss': running_reg / len(test_loader.dataset),
        'accuracy': 100 * correct / total,
        'mse': mse,
        'mae': mae
    }


def test_model(model, test_loader, device, return_predictions=False):
    model.eval()
    all_auc_preds = []
    all_auc_labels = []
    all_loc_preds = []
    all_loc_labels = []

    with torch.no_grad():
        for inputs, (auc_labels, loc_labels) in test_loader:
            inputs = inputs.to(device)
            auc_labels = auc_labels.to(device)
            loc_labels = loc_labels.float().to(device)

            class_out, loc_out = model(inputs)

            # Classification
            _, predicted = torch.max(class_out, 1)
            all_auc_preds.extend(predicted.cpu().numpy())
            all_auc_labels.extend(auc_labels.cpu().numpy())

            # Regression
            all_loc_preds.append(loc_out.cpu().numpy())
            all_loc_labels.append(loc_labels.cpu().numpy())

    # Classification metrics
    auc_accuracy = accuracy_score(all_auc_labels, all_auc_preds)
    precision, recall, f1, avg_tpr, avg_fpr = StatUtils.calculate_metrics(
        np.array(all_auc_labels), np.array(all_auc_preds)
    )
    kappa = StatUtils.calculate_kappa(all_auc_labels, all_auc_preds)

    # Regression metrics
    loc_preds = np.concatenate(all_loc_preds)
    loc_labels = np.concatenate(all_loc_labels)
    mse = mean_squared_error(loc_labels, loc_preds)
    mae = mean_absolute_error(loc_labels, loc_preds)

    results = {
        'auc_accuracy': auc_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'kappa': kappa,
        'avg_tpr': avg_tpr,
        'avg_fpr': avg_fpr,
        'mse': mse,
        'mae': mae
    }

    if return_predictions:
        results['loc_preds'] = loc_preds
        results['loc_labels'] = loc_labels
        results['auc_labels'] = np.array(all_auc_labels)
        results['auc_preds'] = np.array(all_auc_preds)

    return results

def main():
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))
    args = parse_args()
    device, _ = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)

    # Initialize data loader
    if args.range == 'Simulate_0dB_300':
        data_loader = DatasetLoader(
            data_path=f"preprocess/processed_data/Simulate/{args.user}_users/data",
            label_path=f"preprocess/processed_data/Simulate/{args.user}_users/labels",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
    else:
        data_loader = DatasetLoader(
            data_path=f"preprocess/processed_data/OATS/{args.user}_users/data",
            label_path=f"preprocess/processed_data/OATS/{args.user}_users/labels",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
    train_loader, test_loader = data_loader.data_load()



    # --------------------------
    # 新增：Optuna超参数优化
    # --------------------------
    if args.optimize_hparams:  # 新增一个命令行参数控制是否优化
        # 创建研究对象（最大化F1分数）
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)  # 剪枝器
        )
        # 运行优化（100次试验）
        study.optimize(
            lambda trial: objective(trial, args, train_loader, test_loader, device),
            n_trials=100,  # 试验次数
            show_progress_bar=True
        )

        # 输出最优超参数
        print("最优超参数:", study.best_params)
        print("最优F1分数:", study.best_value)

        # 用最优参数更新args，准备最终训练
        best_hparams = study.best_params
        args.lr = best_hparams["lr"]
        args.batch_size = best_hparams["batch_size"]
        args.epochs = best_hparams["epochs"]
        reg_weight = best_hparams["reg_weight"]
        weight_decay = best_hparams["weight_decay"]

    # Initialize model
    model = init_model(args)
    model.to(device)

    # Loss functions
    criterion_class = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    if args.evaluate:
        checkpoint = torch.load(os.path.join('model_pt', args.resume), map_location=device)
        saved_args_dict = checkpoint['args']
        from argparse import Namespace
        saved_args = Namespace(**saved_args_dict)
        model = init_model(saved_args)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)

        test_results = test_model(model, test_loader, device)
        print(f"Test Accuracy: {test_results['auc_accuracy']:.4f}%")
        print(f"MSE: {test_results['mse']:.4f}, MAE: {test_results['mae']:.4f}")
        print(f"Precision: {test_results['precision']:.4f}")
        print(f"Recall (TPR): {test_results['avg_tpr']:.4f}")
        print(f"FPR: {test_results['avg_fpr']:.4f}")
        print(f"Cohen's Kappa: {test_results['kappa']:.4f}")
        StatUtils.save_confusion_matrix(test_results['auc_labels'], test_results['auc_preds'])

        StatUtils.calculate_and_plot_cdf(test_results['loc_preds'], test_results['loc_labels'], args=args)

        return

    # Scheduler
    scheduler = (
        FakeLR(optimizer) if args.scheduler == 'const' else
        WarmUpCosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader),
                                T_warmup=30 * len(train_loader), eta_min=7e-5)
    )

    # CSV header
    header = [
        "Epoch",
        "Train_Total_Loss", "Train_Class_Loss", "Train_Reg_Loss", "Train_Acc",
        "Test_Total_Loss", "Test_Class_Loss", "Test_Reg_Loss", "Test_Acc",
        "Test_MSE", "Test_MAE",
        "Test_Precision", "Test_Recall", "Test_F1",
        "Test_Kappa", "Test_TPR", "Test_FPR"
    ]

    csv_path = f"./storage_csv/{args.shape}/{args.user}/MTL_range{args.range}_{args.model}_cl{args.user}_epo{args.epochs}_bz{args.batch_size}_lr{args.lr}.csv"
    best_accuracy = 0.0
    best_mse = float('inf')
    all_results = []

    final_predictions = None



    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Training
        train_metrics = train_one_epoch(
            model, train_loader, criterion_class, criterion_reg, optimizer, device
        )

        # Evaluation
        test_metrics = evaluate_model(
            model, test_loader, criterion_class, criterion_reg, device
        )

        # Detailed testing
        # 只有在最后一个epoch时才返回预测结果
        return_preds = (epoch == args.epochs - 1)
        test_results = test_model(model, test_loader, device, return_predictions=return_preds)

        if return_preds:
            final_predictions = test_results


        # Save results
        all_results.append([
            epoch + 1,
            train_metrics['total_loss'], train_metrics['class_loss'], train_metrics['reg_loss'],
            train_metrics['accuracy'],
            test_metrics['total_loss'], test_metrics['class_loss'], test_metrics['reg_loss'], test_metrics['accuracy'],
            test_metrics['mse'], test_metrics['mae'],
            test_results['precision'], test_results['recall'], test_results['f1'],
            test_results['kappa'], test_results['avg_tpr'], test_results['avg_fpr']
        ])

        # Save best model
        if args.save_pt:
            best_accuracy, best_mse = save_best_model(
                model, args,
                test_metrics['accuracy'],  # 当前测试准确率
                test_metrics['mse'],  # 当前测试MSE
                best_accuracy,  # 历史最佳准确率
                best_mse,  # 历史最小MSE
                './model_pt'
            )

        # Print epoch summary
        print(f"Train Acc: {train_metrics['accuracy']:.4f} | "
              f"Train Loss: {train_metrics['total_loss']:.4f}  "
              f"(Class: {train_metrics['class_loss']:.4f}, Reg: {train_metrics['reg_loss']:.4f})")
        print(f"Test Acc: {test_metrics['accuracy']:.4f}% | "
              f"Test Loss: {test_metrics['total_loss']:.4f} | "
              f"MSE: {test_metrics['mse']:.4f}, MAE: {test_metrics['mae']:.4f}")

    # 在训练结束后计算CDF
    if final_predictions is not None:
        print("\n" + "=" * 50)
        print("Training completed. Computing CDF analysis...")
        StatUtils.calculate_and_plot_cdf(
            final_predictions['loc_preds'],
            final_predictions['loc_labels'],
            args=args
            )
    # Save final results
    StatUtils.save_epoch_results_to_csv(all_results, csv_path, header)

    # Extract data for plotting
    train_total_losses = [x[1] for x in all_results]
    train_accuracies = [x[4] for x in all_results]

    test_total_losses = [x[5] for x in all_results]
    test_class_losses = [x[6] for x in all_results]
    test_reg_losses = [x[7] for x in all_results]
    test_accuracies = [x[8] for x in all_results]
    test_mse_values = [x[9] for x in all_results]
    test_mae_values = [x[10] for x in all_results]

    # Save plots
    StatUtils.save_loss_accuracy_plots(
        train_total_losses, test_total_losses,
         test_class_losses, test_reg_losses,
        train_accuracies, test_accuracies
    )
    StatUtils.save_mae_mse_plots(test_mse_values, test_mae_values)

if __name__ == "__main__":
    main()