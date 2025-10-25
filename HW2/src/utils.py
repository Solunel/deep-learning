"""
utils.py
这里存放所有通用辅助函数，并设置我们的日志系统。
"""

import logging
import csv
import matplotlib.pyplot as plt

def setup_logger():
    """配置并返回一个日志记录器实例。"""
    logging.basicConfig(
        level=logging.INFO,# 设置日志级别为 INFO，表示 INFO, WARNING, ERROR 级别的日志都会被记录。
        # 设置日志输出的格式。
        # %(asctime)s: 日志记录的时间
        # %(levelname)s: 日志级别（如 INFO, ERROR）
        # %(message)s: 具体的日志信息
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'# 设置时间的显示格式。
    )
    # 获取并返回一个名为 __name__ 的 logger 实例。
    # __name__ 会被替换为当前模块的名字（在这里是 'utils'）。
    return logging.getLogger(__name__)

# 定义一个函数来绘制学习曲线（损失和准确率随 epoch 的变化）。
def plot_learning_curves(history, title=''):
    # 使用 matplotlib 创建一个包含 1 行 2 列子图的图窗。figsize 控制整个图窗的大小。
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # 创建一个 x 轴的序列，代表 epoch 的编号
    epochs = range(1, len(history['train_loss']) + 1)

    # --- 在第一个子图 (ax1) 上绘制损失曲线 ---
    ax1.plot(epochs, history['train_loss'], c='tab:red', label='Train Loss')# 绘制训练损失，'tab:red' 是颜色名
    ax1.plot(epochs, history['dev_loss'], c='tab:cyan', label='Dev Loss')# 绘制验证损失。
    ax1.set_title(f'Loss Curve of {title}')# 设置子图标题。
    ax1.legend()# 显示图例。
    ax1.grid(True)# 显示网格线。

    # --- 在第二个子图 (ax2) 上绘制准确率曲线 ---
    ax2.plot(epochs, history['train_acc'], c='tab:red', label='Train Accuracy')
    ax2.plot(epochs, history['dev_acc'], c='tab:cyan', label='Dev Accuracy')
    ax2.set_title(f'Accuracy Curve of {title}')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()# 自动调整子图布局，防止重叠。
    plt.show()# 显示最终的图窗。


def save_pred(preds, file):
    """将预测数组保存到指定的 CSV 文件中。"""
    with open(file, 'w', newline='') as fp:
        writer = csv.writer(fp)# 创建一个 csv.writer 对象，用于向文件中写入数据。
        writer.writerow(['Id', 'Class'])# 写入 CSV 文件的表头。
        for i, p in enumerate(preds): # 使用 enumerate 遍历预测结果数组，它会同时返回索引 i 和值 p。
            writer.writerow([i, p])# 将每一条预测结果（Id 和 Class）写入新的一行。