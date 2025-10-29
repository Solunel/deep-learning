import csv
import logging
import os
import yaml

logger = logging.getLogger(__name__)

#加载配置文件
def load_config(path):
    with open(path,'r',encoding='utf-8') as f:
        return yaml.safe_load(f)

#初始化环境
def init_env(config):
    #初始化日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    #初始化文件目录
    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'],exist_ok=True)

#保存预测结果
def save_pred(preds,path):
    with open(path,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Category'])
        for idx,p in enumerate(preds):
            writer.writerow([idx,p])