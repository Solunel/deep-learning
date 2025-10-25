"""
config.py
配置管理
"""
import os
import torch

# -- 项目路径配置 --
DATA_ROOT = '../data/'#数据所在的根目录
MODEL_DIR = '../my_models/'#模型所在的根目录
LATEST_MODEL_PATH = os.path.join(MODEL_DIR, 'model_latest.ckpt')#最新的模型路径
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'model_best.ckpt')#最好的模型路径
PREDICTION_FILE = './prediction.csv'#预测文件的路径

# -- 训练超参数 (Hyperparameters) --
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")#设备选择
BATCH_SIZE = 128# 定义每个批次包含多少个数据样本
LEARNING_RATE = 0.001 #学习率
NUM_EPOCHS = 20# 定义总共要对整个数据集进行多少轮完整的训练
VAL_RATIO = 0.2# 定义从原始训练数据中，划分多少比例作为验证集
NUM_WORKERS = 3#定义几个线程读取数据

# -- 模型参数 (Model Parameters) --
INPUT_DIM = 429# 定义模型输入层的维度（特征数量）
OUTPUT_DIM = 39#定义模型输出层的维度
