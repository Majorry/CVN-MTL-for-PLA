from cProfile import label
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import numpy as np
import os
import json
from utils.parser import parse_args
import json

__all__ = ['MyDataset', 'DatasetLoader', 'PreFetcher']

args = parse_args()

class PreFetcher:
# 数据预加载
    def __init__(self, loader):
        self.ori_loader = loader
        self.len = len(loader)
        self.stream = torch.cuda.Stream()
        self.next_input = None

    def preload(self):
        try:
            #这里的next指的是将预加载的数据输入到next方法中，加快计算
            self.next_input = next(self.loader)
            #如果没有预加载数据触发异常
        except StopIteration:
            self.next_input = None
            return
# 创建cuda流上下文管理器，就是允许异步操作的，提高效率
        with torch.cuda.stream(self.stream):
    # 把next_input都设置为GPU运行
            for idx, tensor in enumerate(self.next_input):
                self.next_input[idx] = tensor.cuda(non_blocking=True)

    def __len__(self):
        return self.len

    def __iter__(self):
        self.loader = iter(self.ori_loader)
        self.preload()
        return self

    def __next__(self):
        # 确保当前流（即 torch.cuda.current_stream()）在执行当前操作之前等待 self.stream 完成。
        # 这确保了数据传输不会在 GPU 操作之间造成冲突。
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        if input is None:
            raise StopIteration
        for tensor in input:
            tensor.record_stream(torch.cuda.current_stream())
        self.preload()
        return input

def transform_csi_tensor(tensor):
    """
    对输入的 4D CSI tensor（N, C, H, W）进行行翻转操作：
    对每个样本的 feature map（C×H×W）中的 H 维，按奇数行进行左右翻转（W轴逆序）
    """
    # tensor: [N, C, H, W]
    transformed = tensor.clone()  # 复制，避免原地操作

    _, _, H, _ = transformed.shape
    for row in range(1, H, 2):  # 遍历所有奇数行（第2、4、6...行）
        transformed[:, :, row, :] = torch.flip(transformed[:, :, row, :], dims=[-1])

    return transformed

if args.shape == 'transform':
    class MyDataset(Dataset):
        def __init__(self, data, labels, height, width):
            """
            自定义数据集，用于加载训练或测试数据，并进行CSI奇数行逆序转换。
            """
            super(MyDataset, self).__init__()
            assert len(data) == len(labels["auc_labels"]) == len(labels["loc_labels"]), "数据和标签长度不匹配"

            # 原始 reshape 成图像格式 [N, C, H, W]
            reshaped = torch.tensor(data, dtype=torch.complex64).view(-1, data.shape[1], height, width)

            # 对奇数行做左右翻转
            self.data = transform_csi_tensor(reshaped)

            # 分类标签（长整型）
            self.auc_labels = torch.tensor(labels["auc_labels"], dtype=torch.long)

            # 定位标签（浮点型）
            self.loc_labels = torch.tensor(labels["loc_labels"], dtype=torch.float32)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], (self.auc_labels[idx], self.loc_labels[idx])
else:
    class MyDataset(Dataset):
        def __init__(self, data, labels, height=89, width=92):
            """
            自定义数据集，用于加载训练或测试数据。
            """
            super(MyDataset, self).__init__()
            assert len(data) == len(labels["auc_labels"]) == len(labels["loc_labels"]), "数据和标签长度不匹配"
            self.data = torch.tensor(data, dtype=torch.complex64).view(-1, data.shape[1], height, width)
            # 分类标签转换为 long 类型
            self.auc_labels = torch.tensor(labels["auc_labels"], dtype=torch.long)
            # 定位标签转换为 float 类型（回归任务）
            self.loc_labels = torch.tensor(labels["loc_labels"], dtype=torch.float32)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], (self.auc_labels[idx], self.loc_labels[idx])
class DatasetLoader:
    def __init__(self, data_path, label_path, batch_size, num_workers, pin_memory):
        """
        初始化数据加载器，负责加载训练和测试数据并返回 DataLoader。
        """
        assert os.path.isdir(data_path), f"数据路径不存在: {data_path}"
        assert os.path.isdir(label_path), f"标签路径不存在: {label_path}"

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # 数据重塑目标高度和宽度
        self.height, self.width = 89, 92
        file_name = f'range{args.range}_user{args.user}_{args.data_split}'
        # 加载训练数据和标签
        train_data_path = os.path.join(data_path, f"{file_name}_train_data.npy")
        train_labels_path = os.path.join(label_path, f"{file_name}_train_labels.json")
        assert os.path.isfile(train_data_path), f"找不到文件: {train_data_path}"
        assert os.path.isfile(train_labels_path), f"找不到文件: {train_labels_path}"

        train_data = np.load(train_data_path)
        with open(train_labels_path, 'r') as f:
            train_labels = json.load(f)

        # 加载测试数据和标签
        test_data_path = os.path.join(data_path, f"{file_name}_test_data.npy")
        test_labels_path = os.path.join(label_path, f"{file_name}_test_labels.json")
        assert os.path.isfile(test_data_path), f"找不到文件: {test_data_path}"
        assert os.path.isfile(test_labels_path), f"找不到文件: {test_labels_path}"

        test_data = np.load(test_data_path)
        with open(test_labels_path, 'r') as f:
            test_labels = json.load(f)

        # 初始化训练和测试数据集
        self.train_dataset = MyDataset(train_data, train_labels, self.height, self.width)
        self.test_dataset = MyDataset(test_data, test_labels, self.height, self.width)


    def data_load(self):

        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            pin_memory=self.pin_memory, shuffle=True
        )
        test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            pin_memory=self.pin_memory, shuffle=False
        )

        # 如果启用 pin_memory，则使用 PreFetcher
        if self.pin_memory:
            train_loader = PreFetcher(train_loader)
            test_loader = PreFetcher(test_loader)

        return train_loader, test_loader