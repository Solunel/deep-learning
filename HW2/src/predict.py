# predict.py (预测入口)
# 这个脚本是使用训练好的最佳模型进行预测的唯一入口。

import torch
import logging
from dataset import prep_dataloader
from model import Net
from trainer import Trainer
from utils import save_pred, initialize_environment, load_config, setup_logging

logger = logging.getLogger(__name__)

def main():
    """项目预测流程控制器。"""
    # 1. 配置日志和加载配置。
    setup_logging()
    config = load_config("../config.yaml")

    # 2. 初始化环境和设备。
    initialize_environment(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 3. 准备测试数据。
    logger.info("准备测试数据加载器...")
    test_loader, test_dataset = prep_dataloader(config['paths']['data_root'], 'test', config)

    # 4. 初始化一个与训练时结构相同的模型。
    net = Net(input_dim=test_dataset.dim, output_dim=config['model_params']['output_dim']).to(device)

    # 5. 实例化 Trainer，即使只用它的 predict 方法。
    #    这里 train_data 和 dev_data 传入 None，因为预测时不需要。
    trainer = Trainer(
        net=net,
        train_data=None,
        dev_data=None,
        config=config,
        device=device
    )

    # 6. 执行预测。
    preds = trainer.predict(test_loader)

    # 7. 保存结果。
    if preds is not None:
        save_pred(preds, config['paths']['prediction_file'])

    logger.info("预测流程完成!")

if __name__ == '__main__':
    main()