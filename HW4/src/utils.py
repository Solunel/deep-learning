# src/utils.py

import csv
import json
import logging
import math
from pathlib import Path

import yaml
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# ===================================================================
# 环境与文件处理工具
# ===================================================================

def load_config(config_path="../config.yaml"):
    """
    加载并解析 YAML 格式的配置文件。

    Args:
        config_path (str): .yaml 配置文件的路径。

    Returns:
        dict: 包含所有配置参数的字典。
    """
    # 使用 with open(...) 语法可以确保文件在使用后被自动关闭，即使发生错误。
    with open(config_path, "r", encoding='utf-8') as f:
        # yaml.safe_load 是解析 YAML 的标准且安全的方法。
        return yaml.safe_load(f)


def init_env(config):
    """
    根据配置初始化程序运行环境。
    主要任务是配置日志系统和创建必要的文件夹。
    """
    # 配置 Python 的 logging 模块，让程序在运行时能输出带时间戳和级别的日志信息。
    logging.basicConfig(
        level=logging.INFO,  # INFO 级别及以上的日志都会被显示。
        format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式：时间 - 级别 - 消息。
        datefmt='%Y-%m-%d %H:%M:%S',  # 时间格式。
    )

    # 使用 pathlib 创建文件夹，这是比 os.makedirs 更现代的做法。
    # parents=True: 如果上级目录不存在，会自动创建。
    # exist_ok=True: 如果文件夹已经存在，不会报错。
    Path(config['paths']['model_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['log_dir']).mkdir(parents=True, exist_ok=True)


def save_preds(preds, path):
    """
    将预测结果保存为 Kaggle 要求的 CSV 格式。

    Args:
        preds (list of tuples): 包含 (ID, 预测类别) 的元组列表。
                                例如: [('path/to/test1.pt', 'id10270'), ...]
        path (str): 输出的 .csv 文件路径。
    """
    # 'w' 表示写入模式, newline='' 防止在 Windows 下出现多余的空行。
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        # 修正: writerows 应该用于写入多行，写入表头应该用 writerow。
        # 原代码 writer.writerows(['Id', 'Category']) 会把 'I', 'd', ',', ... 分开写入多行，是错误的。
        writer.writerow(['Id', 'Category'])
        # 修正: 遍历的方式也需要调整以匹配传入的 preds 格式。
        # 原代码 for i, p in preds: 是正确的，因为 preds 的每个元素就是一个 (id, category) 的元组。
        # 为了更清晰，我们直接将整个列表写入。
        writer.writerows(preds)


def load_json(data_root_path, filename):
    """
    一个更通用的函数，用于加载指定目录下的 JSON 文件。
    (此函数是对您原代码的优化，让它更具可复用性)

    Args:
        data_root_path (str or Path): 数据集的根目录路径。
        filename (str): 要加载的 .json 文件名。

    Returns:
        dict: 解析后的 JSON 内容。
    """
    # 将根目录和文件名安全地拼接成一个完整的路径。
    json_path = Path(data_root_path) / filename
    # 打开并解析 JSON 文件。
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ===================================================================
# PyTorch 相关工具
# ===================================================================

def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
):
    """
    创建一个学习率调度器，遵循 "预热 + 余弦衰减" 策略。
    这是训练 Transformer 时非常常用且有效的策略。
    """

    def lr_lambda(current_step):
        """
        这个内部函数是核心，它根据当前步数计算一个学习率的缩放因子。
        """
        # 1. 预热阶段 (Warmup Phase)
        #    在训练的最初几步，让学习率从0线性增长到设定的初始值。
        if current_step < num_warmup_steps:
            # 学习率因子从 0 线性增长到 1。
            return float(current_step) / float(max(1, num_warmup_steps))

        # 2. 余弦衰减阶段 (Cosine Decay Phase)
        #    预热结束后，学习率按照余弦函数的形状平滑地从初始值衰减到0。
        # a. 首先计算衰减阶段的进度（一个从0到1的小数）。
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # b. 使用余弦函数计算缩放因子。公式 0.5 * (1 + cos(pi * progress)) 会让输出从 1 平滑下降到 0。
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    # 使用 PyTorch 的 LambdaLR 调度器，它会用我们上面定义的 lr_lambda 函数来动态调整学习率。
    return LambdaLR(optimizer, lr_lambda, last_epoch)