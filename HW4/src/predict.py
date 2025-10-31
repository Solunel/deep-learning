# src_example/predict.py

import logging
import torch
from dataset import prepare_dataloader
from engine import Engine
from model import Classifier
from utils import init_env, load_config, load_json, save_preds

logger = logging.getLogger(__name__)


def predict():
    """
    主预测函数。
    """
    # 1. 加载配置
    config = load_config("../config.yaml")

    # 2. 初始化环境
    init_env(config)

    # 3. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 4. 准备测试数据
    logger.info("准备测试数据加载器...")
    test_loader, _ = prepare_dataloader(config, 'test')

    # 5. 准备模型和映射
    # 加载说话人映射，我们需要它来将模型输出的数字标签（如 0, 1, 2...）
    # 转换回人类可读的说话人名字（如 "id10270", "id10271"...）。
    speaker_map = load_json(config['paths']['data_root'], "mapping.json")
    n_spks = len(speaker_map['speaker2id'])
    id2speaker = speaker_map['id2speaker']

    # 实例化一个与训练时结构完全相同的模型。
    # 稍后引擎会加载训练好的最佳权重来覆盖这里的随机初始化权重。
    net = Classifier(config['model_params'], n_spks).to(device)

    # 6. 创建并启动引擎
    # 再次实例化引擎，但这次我们只提供测试数据。
    engine = Engine(
        net=net,
        config=config,
        device=device,
        test_data=test_loader
    )

    # 调用引擎的 predict 方法，获取预测结果。
    feat_paths, preds = engine.predict()

    # 7. 处理并保存结果
    # 检查 engine.predict() 是否成功返回了结果（如果没有找到最佳模型，会返回 None）。
    if preds is not None:
        speaker_names = [id2speaker[str(p)] for p in preds]

        # --- 新增的打包步骤 ---
        # 使用 zip 将两个列表合并成一个元组的列表。
        # list() 将 zip 对象转换为一个真正的列表。
        # results 的格式会是: [('path/to/test1.pt', 'id10270'), ('path/to/test2.pt', 'id10888'), ...]
        results = list(zip(feat_paths, speaker_names))

        # --- 修正调用 ---
        # 现在我们只传入两个参数：打包好的 results 和文件路径。
        save_preds(results, config['paths']['output_path'])

    logger.info("预测流程完成!")


if __name__ == '__main__':
    predict()