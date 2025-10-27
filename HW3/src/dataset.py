# dataset.py

# 导入需要的库
import torch
import torchvision.transforms as transforms # PyTorch中专门用于图像变换的模块
from PIL import Image # Pillow库，是Python中处理图片事实上的标准库
from torch.utils.data import DataLoader, ConcatDataset # 从PyTorch导入数据加载工具
from torchvision.datasets import DatasetFolder # 一个通用的、从文件夹加载数据的工具

# 【第1步：定义一个顶层函数】
# 这个函数的作用是接收一个文件路径(path)，然后用Pillow库的Image.open()来打开这张图片。
def image_loader(path):
    """一个简单的图片加载函数，用于替代lambda。"""
    # 把它定义成一个独立的顶层函数，是为了解决在使用多进程(NUM_WORKERS > 0)加载数据时可能出现的序列化（pickle）错误。
    return Image.open(path)


# 定义图像变换（数据预处理）
# `transforms.Compose` 像一个管道，可以把多个处理步骤串联起来。
train_tfm = transforms.Compose([
    # 第一步：将所有输入图片的尺寸统一调整为 128x128 像素。模型要求输入大小是固定的。
    transforms.Resize((128, 128)),
    # 第二步：将图片转换为PyTorch的Tensor格式。
    # Tensor是PyTorch中用于计算的基本数据结构，你可以把它想象成一个可以在GPU上运算的多维数组。
    # 这一步还会自动把像素值从 [0, 255] 的范围，归一化到 [0.0, 1.0] 的范围，这有助于模型训练。
    transforms.ToTensor(),
])

# 验证集和测试集的变换流程。通常，我们不会在验证和测试时做数据增强（如随机翻转、旋转等），
# 所以这里的变换和训练集的基础变换是一样的。
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 这个函数是本文件的核心，负责创建所有的数据加载器。
def prepare_dataloaders(config):
    """
    根据配置创建并返回训练、验证和测试的数据加载器。
    它接收config对象，这样就能使用config.py里定义的所有设置。
    """
    # 构造数据集对象(Dataset)
    # `DatasetFolder` 是一个非常方便的类。它会自动遍历你指定的文件夹（比如 TRAIN_LABELED_DIR），
    # 并根据子文件夹的名称来给图片自动分配标签。
    # 例如，'../data/training/labeled/cat/' 里的所有图片标签就是0，'../data/training/labeled/dog/' 里的就是1。
    # loader=image_loader: 告诉它用我们上面定义的函数来打开图片。
    # extensions="jpg": 指定只加载.jpg后缀的文件。
    # transform=train_tfm: 对加载的每张图片都应用我们上面定义的`train_tfm`变换。
    train_set = DatasetFolder(config.TRAIN_LABELED_DIR, loader=image_loader, extensions="jpg", transform=train_tfm)
    valid_set = DatasetFolder(config.VALID_DIR, loader=image_loader, extensions="jpg", transform=test_tfm)
    unlabeled_set = DatasetFolder(config.TRAIN_UNLABELED_DIR, loader=image_loader, extensions="jpg", transform=train_tfm)
    test_set = DatasetFolder(config.TEST_DIR, loader=image_loader, extensions="jpg", transform=test_tfm)

    # 构造数据加载器(DataLoader)
    # DataLoader在Dataset的基础上，实现了自动批处理(batching)、打乱(shuffling)和多进程加载。
    # 参数1: 要加载的数据集。
    # batch_size: 每批的大小，从config中读取。
    # shuffle=True: 在每个epoch开始前，将数据顺序完全打乱。这对于训练至关重要，可以防止模型学到数据的顺序。
    # num_workers: 使用多少个子进程加载数据，从config中读取。
    # pin_memory=True: 一个优化选项。设为True可以锁住内存，加速数据从CPU到GPU的传输。
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    # 对于测试集，我们通常不希望打乱顺序（shuffle=False），以便每次预测的结果顺序都是固定的。
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False)

    # 返回所有创建好的对象，以便在主程序 `main.py` 中使用。
    return train_loader, valid_loader, test_loader, train_set, unlabeled_set