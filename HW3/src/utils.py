# utils.py

import csv # 导入Python内置的CSV文件处理库。

def save_predictions(predictions, file_path):
    """
    将预测结果列表保存到指定格式的 CSV 文件中。
    :param predictions: 一个包含预测类别（如[0, 5, 2, 10, ...]）的列表。
    :param file_path: 要保存到的文件路径（例如 "./predict.csv"）。
    """
    # `with open(...) as f:` 是Python中推荐的文件操作方式。
    # 它能确保在代码块执行完毕后，文件会被自动、安全地关闭，即使中间发生错误。
    # "w" 表示以写入（write）模式打开文件。如果文件已存在，会覆盖它。
    # newline='' 是为了防止在Windows下写入CSV时，每行末尾多出一个空行。
    with open(file_path, "w", newline='') as f:
        # 首先，写入CSV文件的表头（第一行）。`\n` 是换行符。
        f.write("Id,Category\n")

        # 使用 `enumerate` 函数来遍历预测结果列表。
        # 它会同时返回每个元素的索引 `i` (从0开始) 和值 `pred`。
        for i, pred in enumerate(predictions):
             # 使用f-string将ID和预测类别格式化成 "i,pred" 的字符串，并写入文件。
             # 例如，对于第一个预测结果pred=5，写入的就是 "0,5\n"。
             f.write(f"{i},{pred}\n")
    # 打印一条提示信息，告诉用户预测结果已经成功保存到指定位置。
    print(f"预测结果已保存至: {file_path}")