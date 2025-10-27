"""
trainer.py
包含核心的 Trainer 类，封装了训练、验证、预测和检查点管理的所有逻辑。
这是整个项目的引擎。
"""
import torch
from torch import nn
from tqdm import tqdm
import os
import logging
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class Trainer:
    """
    封装了所有训练相关逻辑的类。
    """
    def __init__(self, net, train_data, dev_data, config, device):
        """
        Trainer 类的构造函数。
        :param net: 要训练的 nn.Module 模型。
        :param train_data: 训练 DataLoader。
        :param dev_data: 验证 DataLoader。
        :param config: 项目配置字典。
        :param device: 训练设备 (torch.device)。
        """
        self.net = net
        self.train_data = train_data
        self.dev_data = dev_data
        self.config = config
        self.device = device

        # 在这里初始化训练所需的所有组件
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config['training_hparams']['learning_rate'])
        self.loss_fn = nn.CrossEntropyLoss()
        # 初始化 TensorBoard 的写入器，日志将保存在 config 指定的目录。
        self.writer = SummaryWriter(self.config['paths']['log_dir'])

        # 初始化用于追踪训练状态的变量
        self.start_epoch = 1
        self.best_acc = 0.0
        self.epochs_no_improve = 0 # 用于早停法的计数器

    def _save_checkpoint(self, epoch, is_best=False):
        """内部方法，用于保存检查点。"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc
        }
        model_dir = self.config['paths']['model_dir']
        latest_model_path = os.path.join(model_dir, 'model_latest.ckpt')
        best_model_path = os.path.join(model_dir, 'model_best.ckpt')

        # 始终保存最新的检查点，以便于中断后恢复训练。
        torch.save(checkpoint, latest_model_path)
        # 只有在验证集上取得更好表现时，才保存最佳模型。
        if is_best:
            torch.save(checkpoint, best_model_path)

    def _load_checkpoint(self):
        """内部方法，用于从最新的检查点恢复训练状态。"""
        latest_model_path = os.path.join(self.config['paths']['model_dir'], 'model_latest.ckpt')
        if not os.path.exists(latest_model_path):
            logger.info("未发现最新检查点, 将从零开始训练。")
            return

        logger.info(f"发现最新检查点，正在加载: {latest_model_path}")
        checkpoint = torch.load(latest_model_path, map_location=self.device)

        # 恢复模型和优化器的状态
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint.get('best_acc', 0.0)

        logger.info(f"恢复完成。将从轮数 {self.start_epoch} 开始训练。已知最佳验证集准确率: {self.best_acc:.4f}")

    def _run_validation(self):
        """内部方法，在验证集上评估模型。"""
        self.net.eval() # 将模型设置为评估模式
        total_loss, total_correct = 0, 0
        # torch.no_grad() 上下文管理器会禁用梯度计算，可以节省内存并加速计算。
        with torch.no_grad():
            for x, y in self.dev_data:
                x, y = x.to(self.device), y.to(self.device)
                output = self.net(x)
                loss = self.loss_fn(output, y)
                _, predicted = torch.max(output.data, 1)
                total_correct += (predicted == y).sum().item()
                total_loss += loss.item() * len(x)
        avg_loss = total_loss / len(self.dev_data.dataset)
        avg_acc = total_correct / len(self.dev_data.dataset)
        return avg_loss, avg_acc

    def fit(self):
        """公开方法，启动完整的训练流程，包含训练、验证、记录和早停。"""
        logger.info("开始训练流程...")
        self._load_checkpoint() # 尝试从检查点恢复

        # 从配置中获取早停参数，使用 .get 提供默认值以增加健壮性。
        early_stopping_enabled = self.config['training_hparams'].get('enable_early_stopping', False)
        patience = self.config['training_hparams'].get('patience', 5)

        for epoch in range(self.start_epoch, self.config['training_hparams']['num_epochs'] + 1):
            self.net.train() # 将模型设置为训练模式
            train_loss, train_correct = 0, 0

            # 使用 tqdm 创建一个智能进度条
            for x, y in tqdm(self.train_data, desc=f"Epoch {epoch}/{self.config['training_hparams']['num_epochs']}"):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad() # 清空上一轮的梯度
                output = self.net(x) # 前向传播
                loss = self.loss_fn(output, y) # 计算损失
                loss.backward() # 反向传播，计算梯度
                self.optimizer.step() # 更新模型权重

                _, predicted = torch.max(output.data, 1)
                train_correct += (predicted == y).sum().item()
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
                break # 跳出训练循环

        logger.info("训练结束。")
        self.writer.close() # 关闭 writer，确保所有日志都已写入文件

    def predict(self, test_data):
        """公开方法，用于在测试集上进行预测。"""
        best_model_path = os.path.join(self.config['paths']['model_dir'], 'model_best.ckpt')
        logger.info(f"加载最佳模型用于预测: {best_model_path}")
        if not os.path.exists(best_model_path):
            logger.error("无法进行预测：未找到最佳模型检查点！")
            return None

        checkpoint = torch.load(best_model_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"最佳模型加载成功 (来自 epoch {checkpoint['epoch']})。")

        self.net.eval()
        outputs = []
        with torch.no_grad():
            for x in tqdm(test_data, desc="Testing"):
                x = x.to(self.device)
                output = self.net(x)
                _, test_pred = torch.max(output, 1)
                outputs.append(test_pred.cpu()) # 将结果移回 CPU
        return torch.cat(outputs, dim=0).numpy()