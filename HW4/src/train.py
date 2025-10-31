# src_example/train.py

import logging
import torch
# 从我们自己编写的模块中导入必要的类和函数
from dataset import prepare_dataloader
from engine import Engine
from model import Classifier
from utils import init_env, load_config, load_json

# 获取日志记录器
logger = logging.getLogger(__name__)


def train():
    config = load_config("../config.yaml")
    init_env(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    logger.info("准备数据加载器...")
    train_loader, _ = prepare_dataloader(config, 'train')  # 获取训练数据
    dev_loader, _ = prepare_dataloader(config, 'dev')  # 获取验证数据

    # 5. 准备模型
    # 加载说话人映射文件以确定类别总数（n_spks）。
    speaker_map = load_json(config['paths']['data_root'],"mapping.json")
    n_spks = len(speaker_map['speaker2id'])
    logger.info(f"发现 {n_spks} 位说话人。")

    # 实例化模型，并将模型参数和类别总数传入。
    # .to(device) 会将模型的所有参数和缓冲区移动到指定的设备（GPU或CPU）上。
    net = Classifier(config['model_params'], n_spks).to(device)

    # 6. 创建并启动引擎
    # 实例化核心引擎，将模型、配置、设备和数据都交给它管理。
    engine = Engine(
        net=net,
        config=config,
        device=device,
        train_data=train_loader,
        dev_data=dev_loader
    )
    # 调用引擎的 run_training 方法，开始漫长的训练过程。
    engine.run_training()

    logger.info("训练流程完成!")
    # 训练结束后，提示用户如何使用 TensorBoard 查看训练曲线。
    logger.info(f"要查看训练过程，请在项目根目录运行: tensorboard --logdir {config['paths']['log_dir']}")


# Python 的标准入口点。
# 当直接运行 `python train.py` 时，__name__ 的值是 "__main__"，因此下面的代码会被执行。
if __name__ == '__main__':
    train()