"""
model.py
模型定义
"""

from torch import nn

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def loss(self, output, target):
        return self.loss_fn(output, target)