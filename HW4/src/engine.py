# src/engine.py

import logging
import os
import torch
# 修正: 神经网络模块 nn 应该直接从 torch 导入，而不是 torch.ao
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 从我们自己的工具箱中导入学习率调度器函数
from utils import get_cosine_schedule_with_warmup

# 获取日志记录器实例
logger = logging.getLogger(__name__)


class Engine:
    """
    核心引擎类，项目的“发动机”。
    它封装了所有复杂的训练、验证、预测和检查点管理的逻辑，
    使得主程序 (train.py, predict.py) 的代码可以保持非常简洁。
    """

    def __init__(self, net, config, device, train_data=None, dev_data=None, test_data=None):
        """
        初始化引擎，就像组装一台发动机，把所有必要的零件都准备好。

        Args:
            net (nn.Module): 要训练或使用的神经网络模型。
            config (dict): 包含所有配置的字典。
            device (torch.device): 计算设备 (CPU or CUDA)。
            train_data, dev_data, test_data (DataLoader): 对应的数据加载器。
        """
        # --- 核心组件 ---
        self.net = net  # 要驱动的模型
        self.config = config  # 发动机的控制参数（配置文件）
        self.device = device  # 发动机的工作平台（CPU 或 GPU）

        # --- 数据管道 ---
        self.train_data = train_data  # 训练数据传送带
        self.dev_data = dev_data  # 验证数据传送带
        self.test_data = test_data  # 测试数据传送带

        # --- 训练相关的“工具” ---
        # 损失函数：用于衡量模型预测与真实标签之间的差距。CrossEntropyLoss 是多分类任务的标准选择。
        self.loss = nn.CrossEntropyLoss()

        # 优化器：负责根据损失计算出的梯度来更新模型权重。AdamW 是 Transformer 的常用选择。
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=config['hparams']['learning_rate'])

        # 学习率调度器：动态调整学习率，帮助模型更好地收敛。
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config['hparams']['warmup_steps'],
            num_training_steps=config['hparams']['total_steps']
        )

        # 可视化日志记录器：将训练过程中的关键指标（如损失、准确率）写入文件，方便用 TensorBoard 查看。
        self.writer = SummaryWriter(config['paths']['log_dir'])

        # --- 状态追踪变量 ---
        self.current_step = 0  # 记录当前训练了多少步
        self.best_acc = 0.0  # 记录验证集上出现过的最高准确率
        self.epochs_no_improve = 0  # 记录验证集准确率连续多少次没有提升（用于早停）

    def _save_checkpoint(self, is_best=False):
        """
        保存“行车记录仪”数据（检查点），方便随时恢复。
        """
        # 将模型、优化器、调度器以及当前训练进度等所有状态打包成一个字典。
        checkpoint = {
            'step': self.current_step,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc
        }
        model_dir = self.config['paths']['model_dir']
        # 始终保存一份最新的记录 (latest_model.pth)。
        torch.save(checkpoint, os.path.join(model_dir, 'latest_model.pth'))
        # 如果这是历史最好成绩，就额外保存一份永久纪念 (best_model.pth)。
        if is_best:
            torch.save(checkpoint, os.path.join(model_dir, 'best_model.pth'))

    def _load_checkpoint(self):
        """
        加载“行车记录仪”数据，实现断点续训。
        """
        latest_model_path = os.path.join(self.config['paths']['model_dir'], 'latest_model.pth')
        if not os.path.exists(latest_model_path):
            logger.info("未发现最新检查点, 将从零开始训练。")
            return

        logger.info(f"发现最新检查点，正在加载: {latest_model_path}")
        # 加载检查点字典
        checkpoint = torch.load(latest_model_path, map_location=self.device)
        # 将字典中的各项状态恢复到引擎的对应组件中。
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_step = checkpoint['step']
        self.best_acc = checkpoint['best_acc']
        logger.info(f"恢复完成。将从步数 {self.current_step} 开始。已知最佳验证集准确率: {self.best_acc:.4f}")

    def _run_validation(self):
        """
        执行一次“随堂测验”（在验证集上评估）。
        """
        self.net.eval()  # 关键！让模型进入“考试模式”，关闭 Dropout 等。
        total_loss, total_correct = 0, 0
        # with torch.no_grad(): 临时关闭梯度计算，节省资源，加速运行。
        with torch.no_grad():
            # 使用 tqdm 创建一个漂亮的进度条。
            for x, y in tqdm(self.dev_data, desc="Validation"):
                x, y = x.to(self.device), y.to(self.device)
                output = self.net(x)
                loss = self.loss(output, y)
                _, predicted = torch.max(output, 1)  # 找到分数最高的那个类别作为预测结果。
                total_correct += predicted.eq(y).sum().item()  # 累加答对的题目数量。
                total_loss += loss.item() * len(x)
        # 计算平均分并返回。
        return total_loss / len(self.dev_data.dataset), total_correct / len(self.dev_data.dataset)

    def run_training(self):
        """
        启动发动机，开始漫长的“训练旅程”。
        """
        logger.info("开始训练流程...")
        self._load_checkpoint()  # 旅程开始前，先看看有没有上次的记录可以接着跑。

        train_iterator = iter(self.train_data)  # 将数据传送带变成一个可迭代对象。

        patience = self.config['hparams']['patience']
        early_stop = self.config['hparams']['enable_early_stopping']

        # 创建一个总里程为 total_steps 的进度条。
        pbar = tqdm(initial=self.current_step, total=self.config['hparams']['total_steps'], desc="Training")
        while self.current_step < self.config['hparams']['total_steps']:
            self.net.train()  # 关键！让模型进入“学习模式”。
            try:
                # 从传送带上取下一批训练材料。
                x, y = next(train_iterator)
            except StopIteration:
                # 如果传送带上的材料用完了（一轮结束），就再装满一轮。
                train_iterator = iter(self.train_data)
                x, y = next(train_iterator)

            x, y = x.to(self.device), y.to(self.device)

            # --- 核心训练六步法 ---
            self.optimizer.zero_grad()  # 1. 大脑清零，准备接收新知识
            output = self.net(x)  # 2. 看题（前向传播），给出答案
            loss = self.loss(output, y)  # 3. 对答案，计算错得有多离谱（计算损失）
            loss.backward()  # 4. 反思错误（反向传播，计算梯度）
            self.optimizer.step()  # 5. 修正认知（更新模型权重）
            self.scheduler.step()  # 6. 调整学习状态（更新学习率）

            # --- 更新仪表盘 ---
            self.current_step += 1
            pbar.update(1)  # 进度条前进一格
            # 在进度条上显示实时油耗（损失）和当前档位（学习率）。
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self.scheduler.get_last_lr()[0]:.6f}")
            # 将油耗记录到 TensorBoard。
            self.writer.add_scalar('Loss/train_step', loss.item(), self.current_step)

            # --- 到达休息站：验证、保存与早停 ---
            # 每跑 valid_steps 公里，就进一次休息站。
            if self.current_step % self.config['hparams']['valid_steps'] == 0:
                dev_loss, dev_acc = self._run_validation()  # 进行一次随堂测验
                logger.info(f"Step {self.current_step}: Val Acc: {dev_acc:.4f}, Val Loss: {dev_loss:.4f}")
                # 将测验成绩记录到 TensorBoard。
                self.writer.add_scalar('Accuracy/validation', dev_acc, self.current_step)
                self.writer.add_scalar('Loss/validation', dev_loss, self.current_step)

                # 如果这次成绩是历史最好
                if dev_acc > self.best_acc:
                    self.best_acc = dev_acc
                    self.epochs_no_improve = 0  # 信心大增，重置“失去耐心”计数器
                    logger.info(f"发现更好的模型! 准确率: {self.best_acc:.4f}")
                    self._save_checkpoint(is_best=True)
                else:
                    self.epochs_no_improve += 1  # 成绩没提升，有点小失望
                    logger.info(f"验证集准确率未提升. 早停计数: {self.epochs_no_improve}/{patience}")

                # 无论成绩如何，都保存一下最新的行车记录。
                self._save_checkpoint(is_best=False)

                # 检查是否已经失去耐心
                if early_stop and self.epochs_no_improve >= patience:
                    logger.info(f"连续 {patience} 次成绩未提升，不想跑了，提前结束旅程。")
                    break  # 跳出训练循环

        pbar.close()
        logger.info("训练结束。")
        self.writer.close()

    def predict(self):
        """
        执行“期末考试”，使用最好的模型进行最终预测。
        """
        best_model_path = os.path.join(self.config['paths']['model_dir'], 'best_model.pth')
        if not os.path.exists(best_model_path):
            logger.error("无法预测：未找到最佳模型！请先训练。")
            return None, None

        logger.info(f"加载最佳模型用于预测: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"模型加载成功 (来自 step {checkpoint['step']}, 验证集准确率 {checkpoint['best_acc']:.4f})。")

        self.net.eval()  # 进入“考试模式”
        all_preds, all_feat_paths = [], []
        with torch.no_grad():
            for feat_paths, mels in tqdm(self.test_data, desc="Predicting"):
                mels = mels.to(self.device)
                output = self.net(mels)  # 模型给出答案
                _, preds = torch.max(output, 1)  # 选出最可能的答案
                all_preds.extend(preds.cpu().numpy())  # 收集答案
                all_feat_paths.extend(feat_paths)  # 收集对应的题号
        # 返回所有题号和对应的答案
        return all_feat_paths, all_preds
