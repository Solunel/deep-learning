import csv
import os
import matplotlib.pyplot as plt
import numpy
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader


# ==================== 数据处理 ====================
# (已修改：增加了 mean 和 std 参数以进行正确的标准化)
class Covid19dataset(Dataset):
    def __init__(self, path, mode='train', mean=None, std=None):
        super().__init__()
        self.mode = mode

        # 读取数据
        with open(path) as file:
            data_csv = list(csv.reader(file))
            data = numpy.array(data_csv[1:])[:, 1:].astype(float)

        # 分割数据
        if mode == 'test':
            # 测试集没有标签
            data = data[:, 0:93]
            self.data = torch.FloatTensor(data)
        else:
            # 训练集和验证集有标签
            target = data[:, -1]
            data = data[:, 0:93]

            # 按照 9:1 的比例划分训练集和验证集
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]

            self.target = torch.FloatTensor(target[indices])
            self.data = torch.FloatTensor(data[indices])

        # --- 标准化修改 ---
        # 使用传入的训练集均值和标准差进行标准化，避免数据泄露
        if mean is not None and std is not None:
            self.data[:, 40:] = (self.data[:, 40:] - mean) / std

        self.dim = self.data.shape[1]

    def __getitem__(self, index):
        if self.mode in ['train', 'dev']:
            return self.data[index], self.target[index]
        else:  # test mode
            return self.data[index]

    def __len__(self):
        return len(self.data)


# ==================== 数据加载 ====================
# (已修改：增加了 mean 和 std 参数)
def prep_dataloader(path, mode, batch_size, mean=None, std=None, n_jobs=0):
    dataset = Covid19dataset(path, mode, mean=mean, std=std)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=n_jobs,
        drop_last=False,
        pin_memory=True  # 当使用GPU时，设置为True可以加速数据转移
    )
    print(f"{mode.capitalize()} dataloader prepared.")
    return dataloader


# ==================== 定义神经网络模型 ====================
class Net(nn.Module):
    def __init__(self, input_dim):
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


# ==================== 训练、验证、测试函数 ====================
# (已修改：增加了 device 参数，并将数据移至 device)
def dev(net, dev_data, device):
    """计算并返回验证集上的平均损失"""
    net.eval()
    total_loss = 0
    for x, y in dev_data:
        # 将数据移至 GPU 或 CPU
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            loss = net.loss(net(x), y)
            total_loss += loss.item() * len(x)  # 乘以 batch size 以获得总损失
    return total_loss / len(dev_data.dataset)  # 除以数据集总长度得到平均损失


# (已修改：增加了 device 参数，并将数据移至 device)
def test(net, test_data, device):
    """计算并返回预测结果数组"""
    net.eval()
    outputs = []
    for x in test_data:
        # 将数据移至 GPU 或 CPU
        x = x.to(device)
        with torch.no_grad():
            output = net(x)
            # 必须先将 Tensor 移回 CPU 才能转换为 NumPy 数组
            outputs.append(output.cpu())
    return torch.cat(outputs, dim=0).numpy()


# (已修改：增加了 device 参数，并将数据移至 device)
def train(net, train_data, dev_data, device):
    max_epoch = 3000
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_loss_curve = []
    dev_loss_curve = []
    min_mse = 1000.0
    early_stop_count = 0

    for epoch in range(1, max_epoch + 1):
        net.train()
        total_train_loss = 0
        for x, y in train_data:
            # 将数据移至 GPU 或 CPU
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = net(x)
            loss = net.loss(output, y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            # 记录每个 step 的 loss，移回 CPU
            train_loss_curve.append(loss.detach().cpu().item())

        # 每个 epoch 结束后，在验证集上评估
        dev_mse = dev(net, dev_data, device)
        dev_loss_curve.append(dev_mse)

        if dev_mse < min_mse:
            min_mse = dev_mse
            print(f"Epoch {epoch}: Saving model with lower dev loss: {min_mse:.4f}")
            torch.save(net.state_dict(), '../checkpoints/hw1/model.pth')
            early_stop_count = 0
        else:
            early_stop_count += 1

        # 早停机制
        if early_stop_count >= 200:
            print(f"Early stopping at epoch {epoch}")
            break

    return train_loss_curve, dev_loss_curve


# ==================== 绘图和保存结果的函数 ====================
def plot_learning_curve(train_loss, dev_loss, title=''):
    total_steps = len(train_loss)
    x_1 = range(total_steps)
    x_2 = numpy.linspace(0, total_steps, len(dev_loss), endpoint=False)  # 更精确的X轴对应
    plt.figure(figsize=(6, 4))
    plt.plot(x_1, train_loss, c='tab:red', label='train')
    plt.plot(x_2, dev_loss, c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.0)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title(f'Learning curve of {title}')
    plt.legend()
    plt.grid(True)
    plt.show()


# (已修改：确保了数据和模型都在正确的 device 上)
def plot_pred(dv_set, model, device, lim=35.):
    model.eval()
    preds, targets = [], []
    for x, y in dv_set:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            # 必须先将 Tensor 移回 CPU 才能进行后续处理
            preds.append(pred.cpu())
            targets.append(y.cpu())

    preds = torch.cat(preds, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()

    plt.figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('Ground Truth Value')
    plt.ylabel('Predicted Value')
    plt.title('Ground Truth v.s. Prediction')
    plt.grid(True)
    plt.show()


def save_pred(preds, file):
    print(f'Saving results to {file}')
    with open(file, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


# ==================== 主流程 ====================
if __name__ == '__main__':
    # 1. 设置设备 (!!! 关键修改 !!!)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 设置目录和路径
    train_path = '../data/hw1/covid.train.csv'
    test_path = '../data/hw1/covid.test.csv'
    model_dir = "../checkpoints/hw1"
    os.makedirs(model_dir, exist_ok=True)

    # 3. 为标准化计算均值和标准差 (!!! 关键修改 !!!)
    with open(train_path) as file:
        data_csv = list(csv.reader(file))
        full_train_data = numpy.array(data_csv[1:])[:, 1:94].astype(float)

    mean = torch.FloatTensor(full_train_data[:, 40:-1]).mean(dim=0)
    std = torch.FloatTensor(full_train_data[:, 40:-1]).std(dim=0)

    # 4. 加载数据
    train_data = prep_dataloader(train_path, 'train', batch_size=135, mean=mean, std=std)
    dev_data = prep_dataloader(train_path, 'dev', batch_size=135, mean=mean, std=std)
    test_data = prep_dataloader(test_path, 'test', batch_size=135, mean=mean, std=std)

    # 5. 设置模型与训练
    net = Net(train_data.dataset.dim).to(device)  # 将模型移至 device
    train_loss, dev_loss = train(net, train_data, dev_data, device)

    # 6. 绘制学习曲线
    plot_learning_curve(train_loss, dev_loss, title='deep model')

    # 7. 加载最好的模型进行预测
    print("Loading the best model for prediction...")
    # 重新初始化模型并移至 device
    net = Net(train_data.dataset.dim).to(device)
    # 加载模型权重，map_location 确保权重被加载到正确的设备上
    best_net_state = torch.load(os.path.join(model_dir, 'model.pth'), map_location=device)
    net.load_state_dict(best_net_state)

    # 8. 绘制预测图并生成测试结果
    plot_pred(dev_data, net, device)
    preds = test(net, test_data, device)

    # 9. 保存结果
    save_pred(preds, '../results/hw1.csv')
    print('All Done!')