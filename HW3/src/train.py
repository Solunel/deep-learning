import logging

import torch

from HW3.src.dataset import prepare_dataloader
from HW3.src.engine import Engine
from HW3.src.model import Net
from HW3.src.utils import init_env, load_config

logger = logging.getLogger(__name__)

def train():
    config = load_config("../config.yaml")
    init_env(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    logger.info("准备数据加载器...")
    train_loader, train_dataset = prepare_dataloader(config, 'train')
    dev_loader, _ = prepare_dataloader(config, 'dev')

    net = Net().to(device)

    engine = Engine(
        net=net,
        config=config,
        device=device,
        train_data=train_loader,
        dev_data=dev_loader
    )
    engine.run_training()

    logger.info("训练流程完成!")
    logger.info(f"要查看训练过程，请在项目根目录运行: tensorboard --logdir {config['paths']['log_dir']}")

if __name__ == '__main__':
    train()