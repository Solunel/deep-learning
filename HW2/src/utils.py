"""
utils.py
这里存放所有不依赖于项目具体逻辑的通用辅助函数。
"""
import logging
import csv
import os
import yaml

# 获取一个以当前模块名 (__name__ 即 "utils") 命名的 logger 实例。
# 这是 logging 模块的最佳实践，可以方便地追踪日志来源。
logger = logging.getLogger(__name__)

def load_config(path="config.yaml"):
    """
    加载并解析 YAML 格式的配置文件。
    :param path: 配置文件的路径。
    :return: 解析后的配置字典。
    """
    with open(path, 'r', encoding='utf-8') as f:
        # 使用 yaml.safe_load 可以安全地加载 YAML 文件，防止执行恶意代码。
        return yaml.safe_load(f)

def setup_logging():
    """
    配置全局的日志记录器 (Logger)。
    这个函数应该在项目入口 (train.py/predict.py) 的最开始被调用一次。
    """
    logging.basicConfig(
        # 设置日志级别为 INFO，意味着只有 INFO, WARNING, ERROR, CRITICAL 级别的日志会被记录。
        level=logging.INFO,
        # 设置日志输出格式：时间 - 级别 - [模块名] - 日志消息
        format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
        # 设置时间的显示格式。
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def initialize_environment(config):
    """
    根据配置初始化项目环境，例如创建必要的文件夹。
    :param config: 项目配置字典。
    """
    # 使用本模块自己的 logger 记录信息。
    logger.info(f"模型检查点将被保存到: {config['paths']['model_dir']}")
    # exist_ok=True 表示如果文件夹已存在，则不会抛出错误。
    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True) # 确保 TensorBoard 日志目录也存在

def save_pred(preds, file):
    """
    将预测结果数组保存到指定的 CSV 文件中。
    :param preds: 包含所有预测标签的 NumPy 数组。
    :param file: 要保存的 CSV 文件路径。
    """
    logger.info(f"正在保存预测结果到 {file}...")
    # 使用 with 语句确保文件操作结束后能被正确关闭。
    with open(file, 'w', newline='') as fp:
        # 创建一个 csv writer 对象。
        writer = csv.writer(fp)
        # 写入 CSV 文件的表头。
        writer.writerow(['Id', 'Class'])
        # 使用 enumerate 同时获取索引和值，写入每一行数据。
        for i, p in enumerate(preds):
            writer.writerow([i, p])