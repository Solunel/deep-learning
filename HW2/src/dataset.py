"""
dataset.py
负责数据的加载和预处理，定义了 Dataset 和 DataLoader。
"""

import os
import numpy
import torch
from torch.utils.data import Dataset, DataLoader

class TIMITDataset(Dataset):
    """
    自定义的 TIMIT 数据集类，继承自 torch.utils.data.Dataset。
    必须实现 __init__, __getitem__, __len__ 三个方法。
    """
    def __init__(self, data_root, mode='train', val_ratio=0.2):
        """
        数据集的初始化函数，负责加载和预处理数据。
        :param data_root: 数据所在的根目录。
        :param mode: 'train', 'dev' (验证) 或 'test' 模式。
        :param val_ratio: 从训练数据中划分多少比例作为验证集。
        """
        super().__init__()
        self.mode = mode

        # 根据不同的模式，执行不同的数据加载逻辑。
        if mode == 'test':
            test_data = numpy.load(os.path.join(data_root, 'test_11.npy'))
            # 将 NumPy 数组转换为 PyTorch Tensor，并指定数据类型为 float。
            self.data = torch.from_numpy(test_data).float()
            self.label = None # 测试集没有标签
        else:
            # 加载完整的训练数据和标签
            train_data = numpy.load(os.path.join(data_root, 'train_11.npy'))
            train_labels = numpy.load(os.path.join(data_root, 'train_label_11.npy'))

            # 根据配置中定义的比例，计算训练集和验证集的分割点索引。
            split_idx = int(train_data.shape[0] * (1 - val_ratio))

            if mode == 'train':
                # 如果是训练模式，取分割点之前的数据。
                self.data = torch.from_numpy(train_data[:split_idx]).float()
                # 标签需要转为 LongTensor，以供交叉熵损失函数使用。
                self.label = torch.LongTensor(train_labels[:split_idx].astype(numpy.int64))
            elif mode == 'dev':
                # 如果是验证模式，取分割点之后的数据。
                self.data = torch.from_numpy(train_data[split_idx:]).float()
                self.label = torch.LongTensor(train_labels[split_idx:].astype(numpy.int64))

        # 获取数据的特征维度，用于后续初始化模型。
        self.dim = self.data.shape[1]

    def __getitem__(self, index):
        """
        根据索引返回一条数据。
        :param index: 数据索引。
        :return: 一条数据样本 (数据, 标签) 或 (数据)。
        """
        if self.mode in ['train', 'dev']:
            return self.data[index], self.label[index]
        else:
            return self.data[index]

    def __len__(self):
        """
        返回数据集的总长度。
        """
        return len(self.data)

def prep_dataloader(data_root, mode, config):
    """
    一个工厂函数，用于创建和配置 DataLoader。
    :param data_root: 数据根目录。
    :param mode: 'train', 'dev' 或 'test'。
    :param config: 项目配置字典。
    :return: (DataLoader 实例, Dataset 实例)。返回 Dataset 是为了方便获取 input_dim 等属性。
    """
    dataset = TIMITDataset(
        data_root,
        mode=mode,
        val_ratio=config['training_hparams']['val_ratio']
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config['training_hparams']['batch_size'],
        # 只有训练集需要打乱顺序，以增加随机性，提高模型泛化能力。
        shuffle=(mode == 'train'),
        # 使用多少个子进程来加载数据，0表示在主进程中加载。大于0可以加速数据加载。
        num_workers=config['training_hparams']['num_workers'],
        # 如果为 True，会将数据加载到 CUDA 的“固定内存”中，可以加速从 CPU 到 GPU 的数据传输。
        pin_memory=True,
        drop_last=False
    )
    return dataloader, dataset