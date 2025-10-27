# main.py

# 导入PyTorch和它的神经网络模块
import torch
import torch.nn as nn

# 从我们自己编写的其他文件中，导入需要的函数或类。
# 这样做让代码结构清晰，每个文件各司其职。
import config
from dataset import prepare_dataloaders
from model import Classifier
from trainer import train_and_validate, test
from utils import save_predictions

def main():
    """项目的主函数，定义了整个项目的执行流程。"""
    # 第1步: 准备数据
    print("正在准备数据加载器...")
    # 调用 dataset.py 中的 `prepare_dataloaders` 函数，它会返回所有需要的数据加载器和数据集对象。
    train_loader, valid_loader, test_loader, train_set, unlabeled_set = prepare_dataloaders(config)
    print("数据准备完成！")

    # 第2步: 初始化模型、损失函数和优化器
    print(f"正在初始化模型，并将其移动到设备: {config.DEVICE}")
    # 从 model.py 中创建我们在Classifier类中定义的模型实例。
    # .to(config.DEVICE) 将模型的所有参数和缓冲区移动到指定的设备（GPU或CPU）上。
    model = Classifier().to(config.DEVICE)
    # 定义损失函数。nn.CrossEntropyLoss 是用于多分类任务的标准损失函数。
    # 它内部已经包含了Softmax操作，所以我们的模型输出（logits）可以直接喂给它。
    criterion = nn.CrossEntropyLoss()
    # 定义优化器。torch.optim.Adam 是一种非常常用且效果很好的优化算法。
    # 它需要知道要优化哪些参数（`model.parameters()`），并设置学习率和权重衰减等超参数。
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # 第3步: 执行训练和验证
    print("开始训练...")
    # 调用 trainer.py 中的主训练函数 `train_and_validate`。
    # 把模型、数据、优化器、损失函数和配置等所有需要的“零件”都传递给它。
    train_and_validate(model, train_loader, valid_loader, optimizer, criterion, config, train_set, unlabeled_set)
    print("训练结束。")

    # 第4步: 在测试集上生成预测结果
    print("正在测试集上生成预测...")
    # 调用 trainer.py 中的 `test` 函数，使用训练好的模型在测试集上进行预测。
    predictions = test(model, test_loader, config)

    # 第5步: 保存预测结果
    # 调用 utils.py 中的 `save_predictions` 函数，将预测结果列表保存到config中指定的CSV文件里。
    save_predictions(predictions, config.PREDICTION_FILE)

# 这是一个Python程序的标准入口点。
if __name__ == '__main__':
    # 当你通过命令行直接运行 `python main.py` 时，`__name__` 的值就是 `'__main__'`。
    # 这部分代码就会被执行。如果这个文件是被其他文件导入(import)的，这部分代码则不会执行。
    # 这是一种良好的编程习惯，可以确保主逻辑只在直接运行时才被调用。
    main()