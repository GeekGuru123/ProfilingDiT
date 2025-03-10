import json
import os
import numpy as np

# 结果文件所在目录
results_dir = "/home/xuran/ProfilingDiT/evaluation_results"

# 目标指标
target_metrics = [
    "subject_consistency", "background_consistency", "motion_smoothness",
    "dynamic_degree", "aesthetic_quality", "imaging_quality"
]

# 存储所有文件的均值
all_files_avg = {}

# 遍历目录下的所有 JSON 文件
for filename in os.listdir(results_dir):
    if filename.endswith(".json"):  # 只处理 JSON 文件
        file_path = os.path.join(results_dir, filename)

        try:
            with open(file_path, "r") as f:
                data = json.load(f)  # 读取 JSON 文件

            file_metrics = {}

            # 遍历 JSON 仅计算目标指标
            for metric in target_metrics:
                if metric in data:
                    values = data[metric]

                    # 处理嵌套列表，确保只提取数值
                    if isinstance(values, list):
                        flattened_values = []
                        for item in values:
                            if isinstance(item, (int, float)):  # 直接是数值
                                flattened_values.append(item)
                            elif isinstance(item, list):  # 可能是嵌套列表
                                flattened_values.extend([x for x in item if isinstance(x, (int, float))])

                        if flattened_values:  # 确保不为空
                            file_metrics[metric] = np.mean(flattened_values)

            if file_metrics:  # 只记录包含有效数据的文件
                all_files_avg[filename] = file_metrics

                # 打印当前文件的均值
                print(f"\nAverage Metrics for {filename}:")
                for metric, avg_value in file_metrics.items():
                    print(f"  {metric}: {avg_value:.6f}")

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

# 计算所有 JSON 文件的整体均值
overall_metrics = {metric: [] for metric in target_metrics}
file_count = len(all_files_avg)

if file_count > 0:
    # 遍历所有文件的均值
    for file_metrics in all_files_avg.values():
        for metric, avg_value in file_metrics.items():
            overall_metrics[metric].append(avg_value)

    # 计算每个指标的全局平均
    overall_avg = {metric: np.mean(values) for metric, values in overall_metrics.items() if values}

    # 输出整体平均值
    print("\n===== Overall Average Across All JSON Files =====")
    for metric, avg_value in overall_avg.items():
        print(f"{metric}: {avg_value:.6f}")
else:
    print("No valid JSON files found.")
import json
import os
import numpy as np

# 结果文件目录


# 目标指标
target_metrics = [
    "subject_consistency", "background_consistency", "motion_smoothness",
    "dynamic_degree", "aesthetic_quality", "imaging_quality"
]

# 存储所有数值
all_values = []

# 遍历 JSON 结果文件
for filename in os.listdir(results_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(results_dir, filename)

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # 遍历所有目标 metric，提取数值
            for metric in target_metrics:
                if metric in data:
                    values = data[metric]

                    # 处理列表（可能是嵌套结构）
                    if isinstance(values, list):
                        for item in values:
                            if isinstance(item, (int, float)):
                                all_values.append(item)
                            elif isinstance(item, list):  # 可能是嵌套列表
                                all_values.extend([x for x in item if isinstance(x, (int, float))])

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

# 计算最终均值
if all_values:
    overall_average = np.mean(all_values)
    print(f"\n===== Overall Average Across All Metrics and Files =====")
    print(f"Overall Average: {overall_average:.6f}")
else:
    print("No valid metric values found.")
