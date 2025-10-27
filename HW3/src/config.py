# config.py

# 导入torch库，我们用它来检查GPU是否可用。
import torch

# -- 路径配置 (Path Configurations) --
# 定义数据文件夹的根路径。这里的"../"表示“上一级目录”。
# 假设你的代码在 'project/src' 文件夹下，那么数据就在 'project/data' 文件夹下。
DATA_DIR = "../data/"
# 使用f-string（f"..."）的格式化方法，方便地将根路径和子路径拼接起来。
# TRAIN_LABELED_DIR 指向的是有标签的训练图片所在的文件夹。
TRAIN_LABELED_DIR = f"{DATA_DIR}training/labeled"
# TRAIN_UNLABELED_DIR 指向无标签的训练图片文件夹。
TRAIN_UNLABELED_DIR = f"{DATA_DIR}training/unlabeled"
# VALID_DIR 指向验证集图片文件夹。
VALID_DIR = f"{DATA_DIR}validation"
# TEST_DIR 指向测试集图片文件夹。
TEST_DIR = f"{DATA_DIR}testing"
# PREDICTION_FILE 定义了最后生成的预测结果要保存的文件名。'./'表示当前目录。
PREDICTION_FILE = "./predict.csv"

# -- 训练超参数 (Training Hyperparameters) --
# 超参数是需要我们手动设置的参数，它们会影响模型的训练效果和速度。

# 检查你的电脑上是否有可用的NVIDIA GPU。PyTorch通过torch.cuda.is_available()来判断。
# 如果有，我们就把设备设置为 "cuda"（表示使用GPU），这样计算会快很多。
# 如果没有，就设置为 "cpu"，使用电脑的中央处理器。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# BATCH_SIZE（批次大小）是指模型在一次训练迭代中“看”多少张图片。
# 比如设置为128，就是每次从数据集中取出128张图片，一起进行计算和更新。
BATCH_SIZE = 128

# NUM_WORKERS（工作进程数）是在准备数据时，可以同时开启多少个子进程来读取图片。
# 设置为大于0的数（比如4）可以大大加快数据加载速度，防止GPU“饿肚子”（等待数据）。
NUM_WORKERS = 4

# NUM_EPOCHS（训练轮数）是指整个训练数据集要被模型完整地学习多少遍。
# 每一轮(Epoch)都意味着模型把所有的训练数据都看了一遍。
NUM_EPOCHS = 2

# LEARNING_RATE（学习率）是控制模型参数更新幅度的关键参数。
# 想象一下你在下山，学习率就是你每一步迈多大。太大了容易“跨过”最低点，太小了下山会很慢。
LEARNING_RATE = 0.0003

# WEIGHT_DECAY（权重衰减）是一种防止模型“过拟合”的技术。
# “过拟合”就像一个学生只会死记硬背，考试时遇到没见过的题型就不会做了。
# 权重衰减通过在损失函数中增加一个惩罚项，让模型的参数值不会变得过大，从而提高模型的泛化能力。
WEIGHT_DECAY = 1e-5

# GRAD_CLIP_NORM（梯度裁剪范数）是防止“梯度爆炸”的技术。
# 在训练过程中，如果梯度（参数更新的方向和幅度）突然变得非常大，训练会变得不稳定。
# 这行代码的意思是，如果计算出的梯度总和（范数）超过了10.0，就把它强行缩减到10.0，保证训练的稳定。
GRAD_CLIP_NORM = 10.0

# -- 半监督学习配置 (Semi-supervised Learning Configuration) --
# 半监督学习是指同时使用有标签和无标签数据进行训练。

# 这是一个总开关，决定是否要启用半监督学习。目前设置为False，表示不启用。
DO_SEMI_SUPERVISED = False
# 如果启用了半监督学习，这个阈值会起作用。
# 模型会对无标签数据进行预测，如果预测的“信心”（置信度）高于0.65，
# 我们就认为这个预测是可靠的，并把它当作一个“伪标签”加入到训练中。
PSEUDO_LABEL_THRESHOLD = 0.65