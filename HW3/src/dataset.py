import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import DatasetFolder


def image_loader(path):
    return Image.open(path).convert("RGB")


class MyDataset(Dataset):
    def __init__(self, config, mode='train'):
        super(MyDataset, self).__init__()
        self.mode = mode

        test_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomRotation(15),  # 随机旋转
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 随机颜色抖动
            transforms.ToTensor(),
        ])

        if mode == 'train':
            self.dataset = DatasetFolder(
                root=os.path.join(config['paths']['data_root'], 'training', 'labeled'),
                loader=image_loader,
                extensions=('.jpg',),
                transform=train_transform
            )
        elif mode == 'dev':
            self.dataset = DatasetFolder(
                root=os.path.join(config['paths']['data_root'], 'validation'),
                loader=image_loader,
                extensions=('.jpg',),
                transform=test_transform
            )
        elif mode == 'test':
            self.dataset = DatasetFolder(
                root=os.path.join(config['paths']['data_root'], 'testing'),
                loader=image_loader,
                extensions=('.jpg',),
                transform=test_transform
            )
        else:
            raise ValueError(f"未知的模式: {mode}。请选择 'train', 'dev', 或 'test'。")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def prepare_dataloader(config, mode='train'):
    dataset = MyDataset(config, mode=mode)
    dataloader = DataLoader(
        dataset,
        batch_size=config['hparams']['batch_size'],
        shuffle=(mode == 'train'),
        num_workers=config['hparams']['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    return dataloader, dataset