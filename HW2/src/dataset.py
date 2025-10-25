"""
dataset.py
负责数据的加载和预处理
"""

import os
import gc # Garbage Collection，用于手动管理内存
import numpy
import torch

from torch.utils.data import Dataset, DataLoader
from config import VAL_RATIO

class TIMITDataset(Dataset):
    def __init__(self, data_root, mode='train'):
        super().__init__()
        self.mode = mode

        # 根据不同的模式，执行不同的数据加载逻辑。
        if mode == 'test':
            test_data = numpy.load(os.path.join(data_root, 'test_11.npy'))
            self.data = torch.from_numpy(test_data).float()
            self.label = None

            del test_data
        else:
            train_data = numpy.load(os.path.join(data_root, 'train_11.npy'))
            train_labels = numpy.load(os.path.join(data_root, 'train_label_11.npy'))

            # 根据 config 文件中定义的比例，计算训练集和验证集的分割点索引。
            split_idx = int(train_data.shape[0] * (1 - VAL_RATIO))

            if mode == 'train':
                # 如果是训练模式，取分割点之前的数据。
                self.data = torch.from_numpy(train_data[:split_idx]).float()
                # 标签需要先从字符串转为整数，再转为 LongTensor，以供交叉熵损失函数使用
                self.label = torch.LongTensor(train_labels[:split_idx].astype(numpy.int64))
            elif mode == 'dev':
                # 如果是验证模式，取分割点之后的数据。
                self.data = torch.from_numpy(train_data[split_idx:]).float()
                self.label = torch.LongTensor(train_labels[split_idx:].astype(numpy.int64))

            del train_data, train_labels

        gc.collect() # 手动调用垃圾回收，建议 Python 立即清理内存
        self.dim = self.data.shape[1]

    def __getitem__(self, index):
        if self.mode in ['train', 'dev']:
            return self.data[index], self.label[index]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)

def prep_dataloader(data_root, mode, batch_size,num_workers):
    dataset = TIMITDataset(data_root, mode=mode)
    dataloader = DataLoader(
        dataset,# 要打包的数据集
        batch_size,
        shuffle=(mode == 'train'),# 只有训练集需要打乱顺序，以增加随机性
        num_workers=num_workers,# 使用多少个子进程来加载数据，0表示在主进程中加载
        pin_memory=True,# 如果为 True，会将数据加载到 CUDA 的“固定内存”中，可以加速从 CPU 到 GPU 的数据传输
        drop_last=False
    )
    # 返回创建好的 dataloader 和 dataset 实例。
    # 返回 dataset 是为了方便主程序获取 input_dim 等数据集自身的属性。
    return dataloader, dataset
