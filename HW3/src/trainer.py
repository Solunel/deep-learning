# trainer.py

# 导入所需库
import torch
import torch.nn as nn
from tqdm.auto import tqdm # 一个非常酷的库，可以在循环时显示一个漂亮的进度条。
from torch.utils.data import DataLoader, ConcatDataset


def get_pseudo_labels(dataset, model, config):
    # 这个函数是为未来的半监督学习准备的占位符，目前没有实现具体功能。
    # 它的理念是：
    # 1. 用已经训练好的模型去预测所有无标签数据。
    # 2. 找出那些模型预测“信心”很足的结果（比如置信度 > 0.65）。
    # 3. 把这些高置信度的预测结果当作“伪标签”，和原始的有标签数据合并，一起用于下一轮训练。
    print("注意: get_pseudo_labels 函数尚未实现!")
    model.train()  # 确保函数退出时，模型能恢复到训练模式。
    return dataset  # 临时直接返回原始数据集，你需要自己实现这部分逻辑。


def train_and_validate(model, train_loader, valid_loader, optimizer, criterion, config, train_set, unlabeled_set):
    """
    执行完整的训练和验证流程。
    """
    # 外层循环，按照在config中设定的总轮数(NUM_EPOCHS)进行训练。
    for epoch in range(config.NUM_EPOCHS):
        current_train_loader = train_loader
        # --- 半监督学习部分 (目前是关闭的) ---
        if config.DO_SEMI_SUPERVISED:
            # 如果开关打开，这里就会调用 get_pseudo_labels 函数，
            # 然后将带伪标签的数据集和原始训练集合并，创建一个新的、更大的数据加载器。
            pass

        # --- 训练阶段 ---
        model.train()  # 非常重要！告诉模型现在进入“训练模式”。
                       # 这会启用一些只在训练时使用的层，比如 Dropout 和 BatchNorm 的参数更新。

        train_loss, train_accs = [], []  # 创建两个空列表，用于记录每个批次的损失和准确率。

        # 使用tqdm包装数据加载器，这样在训练时会显示一个进度条。
        for batch in tqdm(current_train_loader, desc=f"Train Epoch {epoch + 1}/{config.NUM_EPOCHS}"):
            imgs, labels = batch  # 从DataLoader中取出一个批次（batch）的图片(imgs)和标签(labels)。
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)  # 将它们移动到config中设定的设备上（GPU或CPU）。

            logits = model(imgs)  # 1. 前向传播：将图片喂给模型，得到预测的得分(logits)。
            loss = criterion(logits, labels)  # 2. 计算损失：用损失函数(criterion)比较预测得分和真实标签，计算出差距（损失值）。

            optimizer.zero_grad()  # 3. 梯度清零：在反向传播前，必须清除上一轮计算得到的梯度。
            loss.backward()  # 4. 反向传播：根据损失值，自动计算出所有模型参数的梯度。
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)  # 5. 梯度裁剪：防止梯度爆炸。
            optimizer.step()  # 6. 更新参数：优化器(optimizer)根据计算出的梯度，来更新模型的权重。

            # 计算当前批次的准确率。
            # logits.argmax(dim=-1) 找到得分最高的那个类别的索引，作为模型的预测结果。
            # ( ... == labels) 比较预测结果和真实标签是否相等。
            # .float().mean() 将比较结果（True/False）转为1.0/0.0，然后计算平均值，即为准确率。
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            train_loss.append(loss.item())  # 将当前批次的损失值（一个标量）记录下来。
            train_accs.append(acc)  # 记录准确率。

        avg_train_loss = sum(train_loss) / len(train_loss)  # 计算整个epoch的平均训练损失。
        avg_train_acc = sum(train_accs) / len(train_accs)  # 计算平均训练准确率。
        print(f"[ Train | ... ] loss = {avg_train_loss:.5f}, acc = {avg_train_acc:.5f}") # 打印训练结果。

        # --- 验证阶段 ---
        model.eval()  # 非常重要！告诉模型现在进入“评估模式”。
                      # 这会关闭 Dropout，并让 BatchNorm 使用全局统计数据，保证评估结果的确定性。
        valid_loss, valid_accs = [], []

        # `with torch.no_grad():` 是一个上下文管理器，它告诉PyTorch在这个代码块里不要计算梯度。
        # 因为在验证时我们只关心模型的表现，不需要更新参数，这样做可以节省大量计算资源和内存。
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Valid Epoch {epoch + 1}/{config.NUM_EPOCHS}"):
                # 验证过程与训练类似，但没有梯度计算和参数更新的步骤。
                imgs, labels = batch
                imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
                logits = model(imgs)
                loss = criterion(logits, labels)
                acc = (logits.argmax(dim=-1) == labels).float().mean()
                valid_loss.append(loss.item())
                valid_accs.append(acc)

        avg_valid_loss = sum(valid_loss) / len(valid_loss)
        avg_valid_acc = sum(valid_accs) / len(valid_accs)
        print(f"[ Valid | ... ] loss = {avg_valid_loss:.5f}, acc = {avg_valid_acc:.5f}") # 打印验证结果。


def test(model, test_loader, config):
    """
    在测试集上执行预测并返回结果列表。
    """
    model.eval()  # 同样，切换到评估模式。
    predictions = [] # 创建一个空列表来存放所有预测结果。
    with torch.no_grad(): # 测试时也不需要梯度。
        for batch in tqdm(test_loader, desc="Testing"):
            imgs, _ = batch  # 测试集通常没有标签，所以我们用下划线 `_` 来忽略它。
            logits = model(imgs.to(config.DEVICE))  # 模型进行预测，得到得分。

            # 下面这一行是链式操作，将预测结果处理成我们需要的格式：
            # 1. logits.argmax(dim=-1): 得到每个样本预测的类别索引（一个Tensor）。
            # 2. .cpu(): 将这个Tensor从GPU转移到CPU。
            # 3. .numpy(): 将PyTorch Tensor转换为Numpy数组。
            # 4. .tolist(): 将Numpy数组转换为Python列表。
            # 5. predictions.extend(...): 将当前批次的预测结果列表追加到总的predictions列表中。
            predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    return predictions # 返回包含所有预测结果的列表。