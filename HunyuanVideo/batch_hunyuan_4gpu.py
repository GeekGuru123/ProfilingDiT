import subprocess
import argparse
from itertools import cycle
import os
from multiprocessing import Pool

CONDA_ENV = "HunyuanVideo"  # 替换为你的 Conda 环境名称
GPU_IDS = [0, 1, 2, 3]  # 手动指定 4 个 GPU 的 ID
# MODEL_PATH = "/data/nas/xuran/ProfilingDiT/visualization/CogVideo/THUDM/CogVideoX1.5-5b"

def read_prompts(file_path):
    """读取文本文件中的每一行作为一个 prompt"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def run_command(args):
    """构造并运行 cli_demo.py 命令"""
    gpu_id, prompt = args
    command = f"""
    source /data/nas/xuran/miniconda3/etc/profile.d/conda.sh && conda activate {CONDA_ENV} && \
    python3 sample_video.py \
    --video-size 720 1280 \
    --video-length 49 \
    --infer-steps 50 \
    --prompt "{prompt}" \
    --flow-reverse \
    --use-cpu-offload \
    --save-path ./stepwise \
    --seed 42 \
    --model-base "/persistent/app_user_data/models" \
    --dit-weight "/persistent/app_user_data/models/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt" \
    --delta_cache
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # 绑定到指定 GPU
    print(f"Starting process on GPU {gpu_id} with prompt: {prompt}")
    subprocess.run(command, shell=True, env=env, executable="/bin/bash")  # 需要用 bash 执行

def main(prompt_file):
    prompts = read_prompts(prompt_file)
    gpu_cycle = cycle(GPU_IDS)  # 轮流分配给手动指定的 GPU
    task_list = [(next(gpu_cycle), prompt) for prompt in prompts]  # 任务分配

    # 使用进程池并行执行，每个 GPU 负责自己的任务
    with Pool(len(GPU_IDS)) as pool:
        pool.map(run_command, task_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", type=str, default="/data/nas/xuran/ProfilingDiT/prompt_ablation.txt", help="Path to the prompt file")
    args = parser.parse_args()
    
    main(args.prompt_file)
