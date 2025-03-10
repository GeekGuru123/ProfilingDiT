#!/bin/bash

# 支持的维度列表
dimensions=("subject_consistency" "background_consistency" "motion_smoothness" "dynamic_degree" "aesthetic_quality" "imaging_quality")

# 读取传入的 base_path，默认值为 "/home/xuran/ProfilingDiT/HunyuanVideo/results_step_inverse"
base_path=${1:-"/home/xuran/ProfilingDiT/HunyuanVideo/results_step_inv"}

# 遍历每个维度
for dimension in "${dimensions[@]}"; do
    # 构造视频路径
    videos_path="${base_path}"

    # 检查目录是否存在
    if [ ! -d "$videos_path" ]; then
        echo "Skipping $dimension: Directory $videos_path does not exist."
        continue
    fi

    echo "Processing dimension: $dimension"
    echo "Videos path: $videos_path"

    # 运行 VBench 评估
    python /home/xuran/DIT-Cache/benchmark/VBench/evaluate.py \
        --dimension "$dimension" \
        --videos_path "$videos_path" \
        --mode=custom_input

    echo "Evaluation for $dimension completed."
    echo "--------------------------------------"
done

echo "All evaluations completed!"
