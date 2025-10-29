import logging
import os

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

logger = logging.getLogger(__name__)


class Engine:
    def __init__(self, net,config, device, train_data=None,dev_data=None,test_data=None):
        self.net = net
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.config = config
        self.device = device

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config['hparams']['learning_rate'])

        self.writer = SummaryWriter(self.config['paths']['log_dir'])

        self.start_epoch = 1
        self.best_acc = 0.0
        self.epochs_no_improve = 0

    # 保存检查点
    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc
        }
        model_dir = self.config['paths']['model_dir']
        best_model_path = os.path.join(model_dir, 'best_model.pth')
        latest_model_path = os.path.join(model_dir, 'latest_model.pth')

        torch.save(checkpoint, latest_model_path)
        if is_best:
            torch.save(checkpoint, best_model_path)

    # 恢复检查点
    def _load_checkpoint(self):
        latest_model_path = os.path.join(self.config['paths']['model_dir'], 'latest_model.pth')
        if not os.path.exists(latest_model_path):
            logger.info("未发现最新检查点, 将从零开始训练。")
            return

        logger.info(f"发现最新检查点，正在加载: {latest_model_path}")
        checkpoint = torch.load(latest_model_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_acc = checkpoint['best_acc']

        logger.info(f"恢复完成。将从轮数 {self.start_epoch} 开始训练。已知最佳验证集准确率: {self.best_acc:.4f}")

    # 在验证集上评估模型
    def _run_validation(self):
        self.net.eval()
        total_loss, total_correct = 0, 0

        with torch.no_grad():
            for x,y in tqdm(self.dev_data, desc="Validation"):
                x, y = x.to(self.device), y.to(self.device)
                output = self.net(x)
                loss = self.loss(output, y)

                _, predicted = torch.max(output, 1)
                total_correct+=predicted.eq(y).sum().item()
                total_loss+=loss.item()*len(x)

        avg_loss = total_loss / len(self.dev_data.dataset)
        avg_acc = total_correct / len(self.dev_data.dataset)
        return avg_loss, avg_acc

    # 启动完整的训练流程
    def run_training(self):
        logger.info("开始训练流程...")
        self._load_checkpoint()  # 尝试从检查点恢复

        # 从配置中获取早停参数，使用 .get 提供默认值以增加健壮性。
        early_stopping_enabled = self.config['hparams'].get('enable_early_stopping', False)
        patience = self.config['hparams'].get('patience', 5)

        for epoch in range(self.start_epoch, self.config['hparams']['num_epochs'] + 1):
            self.net.train()  # 将模型设置为训练模式
            train_loss, train_correct = 0, 0

            # 使用 tqdm 创建一个智能进度条
            for x, y in tqdm(self.train_data, desc=f"Epoch {epoch}/{self.config['hparams']['num_epochs']}"):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()  # 清空上一轮的梯度
                output = self.net(x)  # 前向传播
                loss = self.loss(output, y)  # 计算损失
                loss.backward()  # 反向传播，计算梯度
                self.optimizer.step()  # 更新模型权重

                _, predicted = torch.max(output, 1)
                train_correct += predicted.eq(y).sum().item()
                train_loss += loss.item() * len(x)

            # --- 每个 Epoch 结束后的操作 ---
            dev_loss_val, dev_acc_val = self._run_validation()
            avg_train_loss = train_loss / len(self.train_data.dataset)
            avg_train_acc = train_correct / len(self.train_data.dataset)

            logger.info(
                f"Epoch {epoch:02d}: Train Acc: {avg_train_acc:.4f} Loss: {avg_train_loss:.4f} | "
                f"Val Acc: {dev_acc_val:.4f} Loss: {dev_loss_val:.4f}"
            )

            # 使用 TensorBoard 记录标量值
            self.writer.add_scalars('Loss', {
                'train': avg_train_loss,
                'validation': dev_loss_val
            }, epoch)

            self.writer.add_scalars('Accuracy', {
                'train': avg_train_acc,
                'validation': dev_acc_val
            }, epoch)

            # --- 早停法与检查点保存逻辑 ---
            if dev_acc_val > self.best_acc:
                self.best_acc = dev_acc_val
                self.epochs_no_improve = 0
                logger.info(f"发现更好的模型! 准确率: {self.best_acc:.4f}")
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.epochs_no_improve += 1
                logger.info(f"验证集准确率未提升. 早停计数: {self.epochs_no_improve}/{patience}")
                self._save_checkpoint(epoch, is_best=False)

            if early_stopping_enabled and self.epochs_no_improve >= patience:
                logger.info(f"连续 {patience} 轮验证集准确率未提升，触发早停。")
                break  # 跳出训练循环

        logger.info("训练结束。")
        self.writer.close()  # 关闭 writer，确保所有日志都已写入文件



    # 在测试集上进行预测
    def predict(self):
        best_model_path = os.path.join(self.config['paths']['model_dir'], 'best_model.pth')

        logger.info(f"加载最佳模型用于预测: {best_model_path}")
        if not os.path.exists(best_model_path):
            logger.error("无法进行预测：未找到最佳模型检查点！请先完成训练。")
            return None

        checkpoint = torch.load(best_model_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        logger.info(
            f"最佳模型加载成功 (来自 epoch {checkpoint['epoch']}, 验证集准确率 {checkpoint['best_acc']:.4f})。")

        self.net.eval()
        outputs = []
        with torch.no_grad():
            for x, _ in tqdm(self.test_data, desc="Predicting"):
                x = x.to(self.device)
                output = self.net(x)
                _, test_pred = torch.max(output, 1)
                outputs.append(test_pred.cpu())

        return torch.cat(outputs, dim=0).numpy()