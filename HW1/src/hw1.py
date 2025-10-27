import csv
import os
import matplotlib.pyplot as plt
import numpy
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader


# 数据处理
class Covid19dataset(Dataset):
    def __init__(self, path, mode='train'):
        super().__init__()
        self.mode = mode

        # 读取数据
        with open(path) as file:
            data_csv = list(csv.reader(file))
            data = numpy.array(data_csv[1:])[:, 1:].astype(float)

        # 分割数据
        if mode == 'test':
            data = data[:, 0:93]
            self.data = torch.FloatTensor(data)
        else:
            target = data[:, -1]
            data = data[:, 0:93]
            train_index = []
            dev_index = []
            for i in range(data.shape[0]):
                if i % 10 != 0:
                    train_index.append(i)
                else:
                    dev_index.append(i)
            if mode == 'train':
                self.target = torch.FloatTensor(target[train_index])
                self.data = torch.FloatTensor(data[train_index, 0:93])
            else:
                self.target = torch.FloatTensor(target[dev_index])
                self.data = torch.FloatTensor(data[dev_index, 0:93])

        self.data[:, 40:] = (self.data[:, 40:] - self.data[:, 40:].mean(dim=0)) / self.data[:, 40:].std(dim=0)
        self.dim = self.data.shape[1]

    def __getitem__(self, index):
        if self.mode == 'train' or self.mode == 'dev':
            return self.data[index], self.target[index]
        else:
            return self.data[index]

    def __len__(self):
        return self.data.shape[0]


# 数据加载
def prep_dataloader(path, mode, batch_size, n_jobs=0):
    dataset = Covid19dataset(path, mode)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=(mode == 'train'),
                            num_workers=n_jobs,
                            drop_last=False,
                            pin_memory=False
                            )
    print(f"{mode}:数据加载完成")
    return dataloader


#定义神经网络模型
class Net(nn.Module):
    def __init__(self,input_dim):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, x):
        return self.net(x).squeeze(1)

    def loss(self, output, target):
        return self.loss_fn(output, target)

#训练、验证、测试函数
def dev(net,dev_data):
    """计算并返回验证集上的平均损失"""
    net.eval()
    total_loss = []

    for x,y in dev_data:
        with torch.no_grad():
            loss = net.loss(net(x), y)
            total_loss.append(loss.item())

    return sum(total_loss)/len(total_loss)

def test(net, test_data):
    """计算并返回预测结果数组"""
    net.eval()
    outputs = []

    for x in test_data:
        with torch.no_grad():
            output = net(x)
            outputs.append(output.cpu())

    return torch.cat(outputs, dim=0).numpy()

def train(net, train_data, dev_data):
    max_epoch = 3000
    epoch = 1
    optimizer = torch.optim.SGD(net.parameters(),lr=0.001, momentum=0.9)
    train_loss = []
    dev_loss = []
    min_mse = 1000
    break_flag = 0
    while epoch < max_epoch:
        net.train()

        for x,y in train_data:
            optimizer.zero_grad()
            loss = net.loss(net(x),y)
            train_loss.append(loss.detach())
            loss.backward()
            optimizer.step()

        dev_mes = dev(net,dev_data)
        if dev_mes < min_mse:
            min_mse = dev_mes
            print(f"保存网络  轮数：{epoch}  损失：{min_mse}")
            torch.save(net.state_dict(), '../checkpoints/hw1/model.pth')
            break_flag = 0
        else:
            break_flag += 1
        dev_loss.append(dev_mes)

        if break_flag > 200: break
        epoch += 1

    return train_loss, dev_loss

# 绘图和保存结果的函数
def plot_learning_curve(train_loss, dev_loss, title=''):
    total_steps = len(train_loss)
    x_1 = range(total_steps)
    x_2 = x_1[::len(train_loss) // len(dev_loss)]
    plt.figure(1, figsize=(6, 4))
    plt.plot(x_1,train_loss,c='tab:red',label='train')
    plt.plot(x_2, dev_loss, c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.0)
    plt.xlabel('steps') # 设置x轴标签
    plt.ylabel('loss')       # 设置y轴标签
    plt.title(f'Learning curve of {title}')  # 设置图表标题
    plt.legend()
    plt.show()

def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.cpu())
                targets.append(y.cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
        plt.figure(2, figsize=(5, 5))
        plt.scatter(targets, preds, c='r', alpha=0.5)
        plt.plot([-0.2, lim], [-0.2, lim], c='b')
        plt.xlim(-0.2, lim)
        plt.ylim(-0.2, lim)
        plt.xlabel('ground truth value')
        plt.ylabel('predicted value')
        plt.title('Ground Truth v.s. Prediction')
        plt.show()

def save_pred(preds, file):
    print(f'保存结果到： {file}')
    with open(file, 'w',newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        # enumerate不仅返回元素本身，还返回元素的索引
        for i, p in enumerate(preds):
            writer.writerow([i, p])

# 设置存储模型的目录
os.makedirs("../checkpoints/hw1", exist_ok=True)

# 加载数据
train_data = prep_dataloader('../data/hw1/covid.train.csv', 'train', batch_size=135)
dev_data = prep_dataloader('../data/hw1/covid.train.csv', 'dev', batch_size=135)
test_data = prep_dataloader('../data/hw1/covid.test.csv', 'test', batch_size=135)

# 设置模型与训练
net = Net(train_data.dataset.dim)
train_loss, dev_loss = train(net, train_data, dev_data)
plot_learning_curve(train_loss, dev_loss,title='deep model')
del net

# 加载最好的模型进行预测
net = Net(train_data.dataset.dim)
best_net = torch.load('../checkpoints/hw1/model.pth', map_location="cpu")
net.load_state_dict(best_net)

plot_pred(dev_data, net, 'cpu')
preds = test(net, test_data)

save_pred(preds, '../results/hw1.csv')
print('All Done!')

