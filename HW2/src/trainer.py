"""
trainer.py
"""
import torch
from tqdm import tqdm # 导入 tqdm 库，它能为任何可迭代对象（如 DataLoader）添加一个智能的进度条。

# 定义验证函数
def dev(net, dev_data, device):
    """在验证集上评估模型，并返回平均损失和准确率。"""
    net.eval()
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        # 遍历验证集的 DataLoader。
        for x, y in dev_data:
            x, y = x.to(device), y.to(device) # 将数据和标签移动到指定的设备（GPU 或 CPU）
            output = net(x)# 前向传播，获取模型的输出 logits
            loss = net.loss(output, y)# 调用模型内部定义的 loss 方法计算损失。
            # 使用 torch.max 找出 logits 中值最大的那个维度的索引，即预测的类别。
            # dim=1 表示在第二个维度（类别维度）上操作。
            _, predicted = torch.max(output.data, 1)
            total_correct += (predicted == y).sum().item() # 累加正确的预测数量
            # 累加批次的总损失。loss.item() 获取损失值，乘以 len(x)（批大小）得到这个批次的总损失。
            total_loss += loss.item() * len(x)

    avg_loss = total_loss / len(dev_data.dataset)# 计算整个验证集的平均损失
    avg_acc = total_correct / len(dev_data.dataset) # 计算整个验证集的平均准确率。
    return avg_loss, avg_acc

# 定义测试函数。
def test(net, test_data, device):
    """在测试集上进行预测，并返回所有预测结果的 NumPy 数组。"""
    net.eval()
    outputs = []
    with torch.no_grad():

        for x in tqdm(test_data, desc="Testing"):# 使用 tqdm 包装 DataLoader 以显示预测进度。
            x = x.to(device)
            output = net(x)
            _, test_pred = torch.max(output, 1)
            outputs.append(test_pred.cpu())# 将预测结果的 Tensor 从 GPU 移回 CPU，以便后续处理（如转换为 NumPy）。
    # 使用 torch.cat 将所有批次的预测结果拼接成一个大的 Tensor，然后转换为 NumPy 数组。
    return torch.cat(outputs, dim=0).numpy()

# 定义主训练函数。
def train(net, train_data, dev_data, config, logger, device, optimizer, start_epoch, best_acc, history):

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        net.train()
        train_loss, train_correct = 0, 0

        for x, y in tqdm(train_data, desc=f"Epoch {epoch}/{config.NUM_EPOCHS}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = net(x)
            loss = net.loss(output, y)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(output.data, 1)
            train_correct += (predicted == y).sum().item()
            train_loss += loss.item() * len(x)

        dev_loss_val, dev_acc_val = dev(net, dev_data, device)
        avg_train_loss = train_loss / len(train_data.dataset)
        avg_train_acc = train_correct / len(train_data.dataset)

        logger.info(
            f"Epoch {epoch:02d}: Train Acc: {avg_train_acc:.4f} Loss: {avg_train_loss:.4f} | "
            f"Val Acc: {dev_acc_val:.4f} loss: {dev_loss_val:.4f}"
        )

        # 【修改】这里不再是初始化，而是向传入的 history 字典中追加记录
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['dev_loss'].append(dev_loss_val)
        history['dev_acc'].append(dev_acc_val)

        # 当前所有状态的检查点字典
        latest_checkpoint = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'history': history,
        }

        # 无条件保存最新的检查点
        torch.save(latest_checkpoint, config.LATEST_MODEL_PATH)

        # 只有当模型表现更好时，才更新最佳检查点
        if dev_acc_val > best_acc:
            best_acc = dev_acc_val
            logger.info(f"发现了更好的模型!其准确率: {best_acc:.4f}")
            # 更新 best_acc 到字典中，然后保存
            latest_checkpoint['best_acc'] = best_acc
            torch.save(latest_checkpoint, config.BEST_MODEL_PATH)

    return history