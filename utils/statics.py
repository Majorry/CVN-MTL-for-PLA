import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score
from utils.parser import parse_args
import pandas as pd


args = parse_args()

class StatUtils:
    @staticmethod
    def save_loss_accuracy_plots(train_total_losses, test_total_losses,
                                 test_class_losses, test_reg_losses,
                                 train_accuracies, test_accuracies,
                                 save_dir=f'plot/{args.shape}/{args.user}'):
        os.makedirs(save_dir, exist_ok=True)
        epochs = range(1, len(train_total_losses) + 1)

        plt.figure(figsize=(12, 5))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_total_losses, 'b-', label='Train Total Loss')
        plt.plot(epochs, test_total_losses, 'r-', label='Test Total Loss')
        plt.plot(epochs, test_class_losses, 'r--', alpha=0.5, label='Test Class Loss')
        plt.plot(epochs, test_reg_losses, 'm--', alpha=0.5, label='Test Reg Loss')
        plt.title('Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, 'b-', label='Train Accuracy')
        plt.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'range{args.range}_{args.model}_{args.user}_lr{args.lr}_bz{args.batch_size}_loss_accuracy_curves.png'))
        plt.close()

    @staticmethod
    def save_mae_mse_plots(test_mse_values, test_mae_values, save_dir=f'plot/{args.shape}/{args.user}'):
        os.makedirs(save_dir, exist_ok=True)
        epochs = range(1, len(test_mse_values) + 1)

        plt.figure(figsize=(10, 5))

        plt.plot(epochs, test_mse_values, 'r-o', label='Test MSE')
        plt.plot(epochs, test_mae_values, 'b--s', label='Test MAE')
        plt.title('Regression Metrics')
        plt.xlabel('Epochs')
        plt.ylabel('Error Value')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'range{args.range}_{args.model}_{args.user}_lr{args.lr}_bz{args.batch_size}_mae_mse_curves.png'))
        plt.close()

    @staticmethod
    def save_confusion_matrix(all_labels, all_preds):
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig(f'plot/{args.shape}/{args.user}_{args.model}_{args.classes}_confusion_matrix.png', dpi=300)  # 设置DPI为300

    @staticmethod
    def calculate_metrics(all_labels, all_preds):
        """计算 Precision、Recall、F1、TPR 和 FPR，避免除零错误"""
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        cm = confusion_matrix(all_labels, all_preds)

        # 计算TPR和FPR，避免除零错误
        tpr = np.diag(cm) / np.sum(cm, axis=1)  # 每类的 TPR
        tpr[np.isnan(tpr)] = 0  # 处理除零产生的NaN值，将其设为0

        fpr = (np.sum(cm, axis=0) - np.diag(cm)) / np.sum(cm)  # 每类的 FPR
        fpr[np.isnan(fpr)] = 0  # 处理除零产生的NaN值，将其设为0

        avg_tpr = np.mean(tpr)
        avg_fpr = np.mean(fpr)

        return precision, recall, f1, avg_tpr, avg_fpr

    @staticmethod
    def calculate_kappa(all_labels, all_preds):
        kappa = cohen_kappa_score(all_labels, all_preds)
        print(f"Cohen's Kappa: {kappa:.4f}")
        return kappa

    @staticmethod
    def save_epoch_results_to_csv(results, file_path, header):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(results)

    @staticmethod
    def measure_time(func):
        """装饰器：测量函数执行时间"""

        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            return result, elapsed_time

        return wrapper

    @staticmethod
    def calculate_and_plot_cdf(loc_preds, loc_labels, save_path=None, args=None):
        """
        计算并绘制回归预测误差的累积分布函数(CDF)，并保存CDF数据

        Parameters:
        - loc_preds: 模型预测的位置值
        - loc_labels: 真实的位置标签
        - save_path: 图片保存路径
        - args: 参数对象，用于生成文件名
        """
        # 计算绝对误差
        absolute_errors = np.abs(loc_preds.flatten() - loc_labels.flatten())
        absolute_errors = absolute_errors[absolute_errors <= 2.0]

        # 创建0到2m的x轴点
        x_range = np.linspace(0, 2.0, 1000)

        # 计算CDF值
        cdf_values = [(np.sum(absolute_errors <= x) / len(absolute_errors)) for x in x_range]
        cdf_values = np.array(cdf_values)

        # 创建图形
        plt.figure(figsize=(10, 6))
        plt.plot(x_range, cdf_values, 'b-', linewidth=2, label='CDF of Absolute Error')
        plt.xlabel('Absolute Error (m)', fontsize=12)
        plt.ylabel('Cumulative Probability', fontsize=12)
        plt.title('Cumulative Distribution Function of Localization Error', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(0, 2.0)
        plt.ylim(0, 1.0)

        # 添加关键点标注
        key_points = [0.5, 1.0, 1.5, 2.0]
        for point in key_points:
            if point <= 2.0:
                cdf_at_point = np.sum(absolute_errors <= point) / len(absolute_errors)
                plt.axvline(x=point, color='red', linestyle='--', alpha=0.5)
                plt.axhline(y=cdf_at_point, color='red', linestyle='--', alpha=0.5)
                plt.text(point + 0.05, cdf_at_point - 0.05,
                         f'P(error≤{point}m)={cdf_at_point:.3f}',
                         fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))

        # 生成保存路径
        if save_path is None:
            if args is not None:
                save_dir = f"plot/{args.shape}/{args.user}/"
                os.makedirs(save_dir, exist_ok=True)
                filename = f"CDF_MTL_range{args.range}_{args.model}_cl{args.user}_epo{args.epochs}_bz{args.batch_size}_lr{args.lr}.png"
                save_path = os.path.join(save_dir, filename)
            else:
                save_path = "./cdf_plot.png"

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # === 保存CDF数据 ===
        temp_dir = "./TEMP"
        os.makedirs(temp_dir, exist_ok=True)

        cdf_data = pd.DataFrame({
            'x_range (m)': x_range,
            'cdf_values': cdf_values
        })

        error_data = pd.DataFrame({
            'absolute_errors': absolute_errors
        })

        # 文件名根据args定义，否则默认
        if args is not None:
            cdf_file = f"CDFData_range{args.range}_{args.model}_cl{args.user}_epo{args.epochs}_bz{args.batch_size}_lr{args.lr}.csv"
            err_file = f"Errors_range{args.range}_{args.model}_cl{args.user}_epo{args.epochs}_bz{args.batch_size}_lr{args.lr}.csv"
        else:
            cdf_file = "CDFData.csv"
            err_file = "Errors.csv"

        cdf_data.to_csv(os.path.join(temp_dir, cdf_file), index=False)
        error_data.to_csv(os.path.join(temp_dir, err_file), index=False)

        # 打印信息
        print(f"\n=== CDF Analysis Results ===")
        print(f"Total samples: {len(absolute_errors)}")
        print(f"Mean absolute error: {np.mean(absolute_errors):.4f}m")
        print(f"Median absolute error: {np.median(absolute_errors):.4f}m")
        print(f"90th percentile error: {np.percentile(absolute_errors, 90):.4f}m")
        print(f"95th percentile error: {np.percentile(absolute_errors, 95):.4f}m")

        for point in key_points:
            if point <= 2.0:
                cdf_at_point = np.sum(absolute_errors <= point) / len(absolute_errors)
                print(f"P(error ≤ {point}m) = {cdf_at_point:.4f}")

        print(f"CDF plot saved to: {save_path}")
        print(f"CDF data saved to: {os.path.join(temp_dir, cdf_file)}")
        print(f"Error data saved to: {os.path.join(temp_dir, err_file)}")

        return x_range, cdf_values, absolute_errors