import logging
import os

import torch

from HW3.src.dataset import prepare_dataloader
from HW3.src.engine import Engine
from HW3.src.model import Net
from HW3.src.utils import init_env, load_config, save_pred

logger = logging.getLogger(__name__)


def predict():
    config = load_config("../config.yaml")
    init_env(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    logger.info("准备数据加载器...")
    test_loader, test_dataset = prepare_dataloader(config, 'test')

    net = Net().to(device)

    engine = Engine(
        net=net,
        config=config,
        device=device,
        test_data=test_loader
    )
    preds = engine.predict()
    if preds is not None:
        save_pred(preds, '../predict.csv')

    logger.info("预测流程完成!")


if __name__ == '__main__':
    predict()