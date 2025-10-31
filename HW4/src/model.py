# src/model.py

import torch.nn as nn


class Classifier(nn.Module):
    """
    基于 Transformer Encoder 的说话人分类器。
    这个类定义了我们AI保安的“大脑结构”。
    """

    def __init__(self, model_params, n_spks):
        """
        大脑的建造过程：在这里预先创建好所有的神经元层（零件）。
        """
        # 必须先调用父类 nn.Module 的构造函数
        super().__init__()

        # 从配置中获取模型的核心维度
        d_model = model_params['d_model']

        # --- 部门1: Prenet (接收与格式化部门) ---
        # 创建一个全连接层，负责将输入的40维特征“翻译”成模型内部的 d_model 维。
        self.prenet = nn.Linear(40, d_model)

        # --- 部门2: Transformer Encoder (核心分析部门) ---
        # a. 先定义一个“标准分析单元”的图纸 (TransformerEncoderLayer)。
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # 内部工作维度
            nhead=model_params['nhead'],  # 多头注意力的“头”数
            dim_feedforward=model_params['dim_feedforward'],  # 内部前馈网络的尺寸
            dropout=0.1,  # 训练时随机失活一些神经元，防止过拟合
            batch_first=True  # 让数据维度顺序为 (批次, 序列, 特征)，更方便
        )
        # b. 用这个图纸，建造一个由多层分析单元堆叠而成的“深度分析部门” (TransformerEncoder)。
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=model_params['num_layers']
        )

        # --- 部门3: Prediction Layer (决策与输出部门) ---
        # 使用 nn.Sequential 将多个层按顺序打包成一个流水线。
        self.pred_layer = nn.Sequential(
            # a. 一个全连接层，用于特征提炼
            nn.Linear(d_model, d_model),
            # b. ReLU 激活函数，增加非线性表达能力
            nn.ReLU(),
            # c. 最终的输出层，将特征映射到每个说话人类别的得分上
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        """
        定义数据在大脑中的“工作流程”。

        Args:
            mels (Tensor): 输入的一批声音信号，形状为 (N, L, 40)。

        Returns:
            out (Tensor): 最终的评分列表，形状为 (N, n_spks)。
        """
        # 流程1: 数据进入 Prenet，形状从 (N, L, 40) 变为 (N, L, d_model)。
        out = self.prenet(mels)

        # 流程2: 进入 Encoder 进行深度分析，形状不变，但内容已是深度融合上下文的特征。
        out = self.encoder(out)

        # 流程3: 对分析结果进行总结（平均池化），将可变长度的序列信息浓缩成一个固定长度的向量。
        #         形状从 (N, L, d_model) 变为 (N, d_model)。
        stats = out.mean(dim=1)

        # 流程4: 将总结报告送入决策部门，得到最终的分类评分。
        #         形状从 (N, d_model) 变为 (N, n_spks)。
        out = self.pred_layer(stats)

        # 返回最终结果
        return out