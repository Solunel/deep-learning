"""
model.py
模型定义。
这个文件只负责定义神经网络的结构和前向传播逻辑。
"""

from torch import nn

class Net(nn.Module):
    """
    一个简单的全连接神经网络 (MLP)。
    """
    def __init__(self, input_dim, output_dim):
        """
        模型结构的初始化。
        :param input_dim: 输入特征的维度。
        :param output_dim: 输出类别的数量。
        """
        super(Net, self).__init__()
        # nn.Sequential 是一个容器，网络层将按照定义的顺序依次执行。
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        """
        定义模型的前向传播逻辑。
        :param x: 输入的 Tensor。
        :return: 模型的输出 Tensor (logits)。
        """
        return self.net(x)