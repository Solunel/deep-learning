# src/dataset.py

import torch
import random
from pathlib import Path
from torch.utils.data import Dataset, random_split, DataLoader
# 修正: 假设 utils.py 在同一 src 目录下，导入路径应为 from utils import load_json
from utils import load_json
from torch.nn.utils.rnn import pad_sequence


# ===================================================================
# 数据集定义
# ===================================================================

class TrainDataset(Dataset):
    """
    用于准备【训练】和【验证】数据的数据集类。
    它的核心职责是：
    1. 在初始化时，解析元数据，建立一个包含所有训练样本路径和标签的“任务清单”。
    2. 在被调用时 (__getitem__)，根据索引加载单个样本，并进行随机数据增强。
    """

    def __init__(self, config):
        """
        初始化方法，只在创建实例时运行一次。
        """
        # --- 基础信息记录 ---
        self.config = config
        self.data_root = self.config['paths']['data_root']
        # 从配置中读取训练时随机裁剪的目标长度。
        self.segment_len = self.config['hparams']['segment_len']

        # --- 解析元数据，构建任务清单 ---
        # 1. 加载“名字->编号”映射表 (mapping.json)。
        speaker2id = load_json(self.data_root, "mapping.json")["speaker2id"]

        # 2. 加载“训练计划表” (metadata.json)。
        metadata = load_json(self.data_root, "metadata.json")["speakers"]

        # 3. 创建一个空的“任务清单” (self.data)。
        self.data = []
        # 遍历计划表中的每一位说话人 (speaker) 和他所有的语音 (utterances)。
        for speaker, utterances in metadata.items():
            # 再次遍历这位说话人的每一段语音。
            for utterance in utterances:
                # 将 [文件路径, 对应的数字编号] 这条任务，添加到任务清单中。
                self.data.append([utterance["feature_path"], speaker2id[speaker]])

    def __len__(self):
        """
        返回数据集中样本的总数。
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        根据索引 (index)，获取并返回一个经过处理的训练样本。
        """
        # 从任务清单中找到对应的任务信息。
        feat_path, speaker_id = self.data[index]

        # 根据路径加载真实的语音数据 (.pt 文件)。
        mel = torch.load(Path(self.config['paths']['data_root']) / feat_path)

        # 数据增强：如果语音片段太长，就随机截取一段。
        if len(mel) > self.segment_len:
            start = random.randint(0, len(mel) - self.segment_len)
            mel = mel[start:start + self.segment_len]

        # 无论是否裁剪，都将数字标签转换为 PyTorch Tensor。
        speaker_id = torch.tensor(speaker_id).long()

        # 返回准备好的训练材料：(处理好的语音, 对应的标签)。
        return mel.float(), speaker_id


class TestDataset(Dataset):
    """
    用于准备【测试】数据的数据集类。
    它的职责非常专一：加载匿名的测试样本。
    """

    def __init__(self, config):
        self.config = config
        # 直接加载“期末考试卷” (testdata.json)，并取出所有考题。
        self.data = load_json(self.config['paths']['data_root'], "testdata.json")["utterances"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 获取第 index 道考题的信息。
        utterance = self.data[index]
        feat_path = utterance["feature_path"]  # 考题的ID（路径）
        # 加载完整的考题语音内容。
        mel = torch.load(Path(self.config['paths']['data_root']) / feat_path)
        # 返回考题材料：(考题ID, 考题内容)。
        return feat_path, mel.float()


# ===================================================================
# 批次整理函数 (Collate Functions)
# ===================================================================

def collate_batch(batch):
    """
    将多个【训练/验证】样本打包成一个批次。
    主要解决两个问题：1. 数据格式转换；2. 长度不一。
    """
    # 1. 分拣：将一批样本中的 mel 和 speaker_id 分开。
    mels, speaker_ids = zip(*batch)
    # 2. 填充：将长度不一的 mel 序列填充到一样长。
    mels = pad_sequence(mels, batch_first=True, padding_value=-20)
    # 3. 堆叠：将零散的 speaker_id Tensor 堆叠成一个 Tensor。
    return mels, torch.stack(speaker_ids)


def test_collate_batch(batch):
    """
    将多个【测试】样本打包成一个批次。
    """
    # 1. 分拣：将 feat_paths 和 mels 分开。
    feat_paths, mels = zip(*batch)
    # 2. 填充：逻辑同上。
    mels = pad_sequence(mels, batch_first=True, padding_value=-20)
    # 3. 返回：feat_paths 是普通 Python 列表，mels 是一个大 Tensor。
    return feat_paths, mels


# ===================================================================
# 数据加载器工厂
# ===================================================================

def prepare_dataloader(config, mode='train'):
    """
    一个工厂函数，根据指定的模式（'train', 'dev', 'test'）创建并返回相应的 DataLoader。
    这是整个数据模块对外的唯一接口，极大地简化了主程序的调用。
    """
    if mode == 'train' or mode == 'dev':
        # 1. 实例化训练数据集。
        dataset = TrainDataset(config)
        # 2. 按 9:1 的比例，动态地将数据集划分为训练集和验证集。
        train_len = int(0.9 * len(dataset))
        lengths = [train_len, len(dataset) - train_len]
        train_set, dev_set = random_split(dataset, lengths)

        # 3. 根据模式，选择本次要加载的数据集部分。
        target_dataset = train_set if mode == 'train' else dev_set

        # 4. 创建并配置 DataLoader。
        dataloader = DataLoader(
            target_dataset,
            batch_size=config['hparams']['batch_size'],
            shuffle=(mode == 'train'),  # 训练时打乱数据
            num_workers=config['hparams']['num_workers'],  # 多进程加速
            pin_memory=True,  # 加速 GPU 数据传输
            drop_last=(mode == 'train'),  # 训练时丢弃最后一个不完整的批次
            collate_fn=collate_batch  # 指定使用我们的训练打包函数
        )
        return dataloader, target_dataset

    elif mode == 'test':
        # 1. 实例化测试数据集。
        dataset = TestDataset(config)

        # 2. 创建并配置 DataLoader。
        dataloader = DataLoader(
            dataset,
            batch_size=config['hparams']['batch_size'],
            shuffle=False,  # 测试时绝不打乱数据
            num_workers=config['hparams']['num_workers'],
            pin_memory=True,
            collate_fn=test_collate_batch  # 指定使用我们的测试打包函数
        )
        return dataloader, dataset
    else:
        # 如果模式无效，则报错。
        raise ValueError(f"未知的模式: {mode}")