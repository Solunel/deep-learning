# main.py

import os
import torch
import config

from dataset import prep_dataloader
from model import Net
from trainer import train, test
from utils import setup_logger, plot_learning_curves, save_pred


# ---辅助函数 1: 初始化环境 ---
def _initialize(logger):
    """设置日志、创建目录等初始化工作。"""
    logger.info(f"使用设备: {config.DEVICE}")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    logger.info(f"模型将被保存到: {config.MODEL_DIR}")


def _load_checkpoint(logger, path):
    """加载指定的检查点文件，如果不存在则返回 None。"""
    if not os.path.exists(path):
        return None

    logger.info(f"发现检查点，正在加载: {path}")
    checkpoint = torch.load(path, map_location=config.DEVICE)
    return checkpoint

# ---辅助函数 2: 准备数据加载器 ---
def _prepare_dataloaders(logger):
    """创建并返回所有的数据加载器。"""
    logger.info("准备数据加载器...")
    train_loader, train_dataset = prep_dataloader(config.DATA_ROOT, 'train', config.BATCH_SIZE, config.NUM_WORKERS)
    dev_loader, _ = prep_dataloader(config.DATA_ROOT, 'dev', config.BATCH_SIZE, config.NUM_WORKERS)
    test_loader, _ = prep_dataloader(config.DATA_ROOT, 'test', config.BATCH_SIZE, config.NUM_WORKERS)
    return train_loader, dev_loader, test_loader, train_dataset.dim


# ---辅助函数 3: 准备训练组件 (模型, 优化器, 检查点) ---
def _prepare_training_components(logger, input_dim):
    """初始化模型、优化器，并智能地从最新的检查点加载状态。"""
    logger.info("初始化模型和优化器...")
    net = Net(input_dim=input_dim, output_dim=config.OUTPUT_DIM).to(config.DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    start_epoch = 1
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'dev_loss': [], 'dev_acc': []}

    # 【修改】优先从 LATEST_MODEL_PATH 恢复训练
    checkpoint = _load_checkpoint(logger, config.LATEST_MODEL_PATH)
    if checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        history = checkpoint['history']
        logger.info(f"恢复完成。当前轮数： {start_epoch}. 已知最佳验证集准确率: {best_acc:.4f}")
    else:
        logger.info("未发现最新检查点,将从零开始训练.")

    return net, optimizer, start_epoch, best_acc, history


# --- 辅助函数 4: 运行最终预测 ---
def _run_prediction(logger, test_loader, input_dim):
    """加载最佳模型，执行预测，并保存结果。"""
    logger.info("加载最佳模型中...")
    pred_net = Net(input_dim=input_dim, output_dim=config.OUTPUT_DIM).to(config.DEVICE)

    # 【修改】只从 BEST_MODEL_PATH 加载用于预测的模型
    checkpoint = _load_checkpoint(logger, config.BEST_MODEL_PATH)
    if not checkpoint:
        logger.error("无法进行预测：未找到表现最佳的模型检查点 (best model checkpoint)！")
        return

    pred_net.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"加载完成，该模型在第 {checkpoint['epoch']} 轮取得最佳验证集准确率： {checkpoint['best_acc']:.4f}")

    logger.info("生成预测中...")
    preds = test(pred_net, test_loader, config.DEVICE)
    save_pred(preds, config.PREDICTION_FILE)
    logger.info(f"预测结果保存于： {config.PREDICTION_FILE}")


# --- 【重构后】的主函数 ---
def main():
    """项目主流程控制器。"""
    # 1. 初始化
    logger = setup_logger()
    _initialize(logger)

    # 2. 准备数据
    train_loader, dev_loader, test_loader, input_dim = _prepare_dataloaders(logger)

    # 3. 准备训练
    net, optimizer, start_epoch, best_acc, history = _prepare_training_components(logger, input_dim)

    # 4. 执行训练
    logger.info("开始训练")
    history = train(net, train_loader, dev_loader, config, logger, config.DEVICE, optimizer, start_epoch, best_acc,
                    history)
    logger.info("训练结束.")

    # 5. 绘制结果
    plot_learning_curves(history, title='Phoneme Classification')

    # 6. 执行预测
    _run_prediction(logger, test_loader, input_dim)

    logger.info("结束!")


if __name__ == '__main__':
    main()