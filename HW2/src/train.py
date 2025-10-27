# train.py (训练入口)
# 这个脚本是启动模型训练的唯一入口。

import torch
import logging
from dataset import prep_dataloader
from model import Net
from trainer import Trainer
from utils import initialize_environment, load_config, setup_logging

logger = logging.getLogger(__name__)

def main():
    """项目主训练流程控制器。"""
    # 1. 配置日志系统，这是程序启动后应该做的第一件事。
    setup_logging()

    # 2. 加载配置文件，所有超参数和路径都从这里读取。
    config = load_config("../config.yaml")

    # 3. 初始化环境，例如创建文件夹。
    initialize_environment(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 4. 准备数据加载器。
    logger.info("准备数据加载器...")
    train_loader, train_dataset = prep_dataloader(config['paths']['data_root'], 'train', config)
    dev_loader, _ = prep_dataloader(config['paths']['data_root'], 'dev', config)

    # 5. 初始化模型，输入维度从数据集中动态获取。
    net = Net(input_dim=train_dataset.dim, output_dim=config['model_params']['output_dim']).to(device)

    # 6. 实例化 Trainer，将所有组件（模型、数据、配置、设备）注入。
    trainer = Trainer(
        net=net,
        train_data=train_loader,
        dev_data=dev_loader,
        config=config,
        device=device
    )

    # 7. 启动训练。
    trainer.fit()

    logger.info("训练流程完成!")
    logger.info(f"要查看训练过程，请在项目根目录运行: tensorboard --logdir {config['paths']['log_dir']}")

if __name__ == '__main__':
    main()