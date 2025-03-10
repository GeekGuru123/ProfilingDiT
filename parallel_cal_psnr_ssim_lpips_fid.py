import os
import cv2
import numpy as np
import shutil
import uuid
import torch
import lpips
import argparse
from sewar import psnr, ssim
from pytorch_fid import fid_score
from tqdm import tqdm
from multiprocessing import Pool
from itertools import cycle

def read_video_frames(video_path):
    """读取视频帧并返回 numpy 数组"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 转换为 RGB 格式
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames)

def process_video(args):
    """
    处理单个视频：计算 PSNR、SSIM、LPIPS 和 FID
    参数 args 为一个元组：(video_name, ref_path, cmp_path, gpu_id)
    """
    video_name, ref_path, cmp_path, gpu_id = args
    print(f"Processing video '{video_name}' on GPU {gpu_id}")
    
    # 指定设备（对应 GPU）
    device = torch.device(f"cuda:{gpu_id}")
    
    # 在当前进程中初始化 LPIPS 模型（绑定到指定设备）
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    
    # 读取参考视频和待评估视频
    ref_vid = read_video_frames(ref_path)
    cmp_vid = read_video_frames(cmp_path)
    
    # 确保视频帧数一致
    min_len = min(len(ref_vid), len(cmp_vid))
    ref_vid = ref_vid[:min_len]
    cmp_vid = cmp_vid[:min_len]
    
    # 存储各项指标的列表
    psnr_values = []
    ssim_values = []
    lpips_values = []
    
    # 为当前视频生成唯一的临时目录（用于FID计算）
    temp_dir = f"temp_{video_name}_{uuid.uuid4().hex}"
    ref_temp = os.path.join(temp_dir, "ref")
    gen_temp = os.path.join(temp_dir, "gen")
    os.makedirs(ref_temp, exist_ok=True)
    os.makedirs(gen_temp, exist_ok=True)
    
    for i in tqdm(range(min_len), desc=f"Video {video_name}"):
        # 计算 PSNR 和 SSIM
        psnr_val = psnr(ref_vid[i], cmp_vid[i])
        ssim_val, _ = ssim(ref_vid[i], cmp_vid[i])
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        
        # 计算 LPIPS：先将图片转换为 tensor，并传到指定 GPU
        img1 = lpips.im2tensor(ref_vid[i]).to(device)
        img2 = lpips.im2tensor(cmp_vid[i]).to(device)
        lpips_val = loss_fn_alex(img1, img2).item()
        lpips_values.append(lpips_val)
        
        # 保存图片到临时文件夹（FID 需要的输入）
        ref_img_path = os.path.join(ref_temp, f'{i:05d}.png')
        gen_img_path = os.path.join(gen_temp, f'{i:05d}.png')
        cv2.imwrite(ref_img_path, cv2.cvtColor(ref_vid[i], cv2.COLOR_RGB2BGR))
        cv2.imwrite(gen_img_path, cv2.cvtColor(cmp_vid[i], cv2.COLOR_RGB2BGR))
    
    # 计算 FID（注意：此处传入的 device 为当前进程的 GPU）
    fid_value = fid_score.calculate_fid_given_paths([ref_temp, gen_temp], 256, device, 2048, num_workers=0)

    
    # 清理临时文件夹
    shutil.rmtree(temp_dir)
    
    metrics = {
        'video': video_name,
        'PSNR': np.mean(psnr_values),
        'SSIM': np.mean(ssim_values),
        'LPIPS': np.mean(lpips_values),
        'FID': fid_value
    }
    
    print(f"Finished video '{video_name}': PSNR={metrics['PSNR']:.2f}, "
          f"SSIM={metrics['SSIM']:.3f}, LPIPS={metrics['LPIPS']:.3f}, FID={metrics['FID']:.1f}")
    return metrics

def main(ref_dir, cmp_dir, num_gpus=8):
    """
    ref_dir: 参考视频目录
    cmp_dir: 对比视频目录
    num_gpus: 使用的 GPU 数量（并行进程数）
    """
    # 构建参考视频和对比视频文件字典（文件名不含扩展名）
    ref_files = {os.path.splitext(f)[0]: os.path.join(ref_dir, f)
                 for f in os.listdir(ref_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))}
    cmp_files = {os.path.splitext(f)[0]: os.path.join(cmp_dir, f)
                 for f in os.listdir(cmp_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))}
    
    common_files = set(ref_files.keys()) & set(cmp_files.keys())
    print(f"Found {len(common_files)} common video files.")
    
    # 构建任务列表，同时分配 GPU（轮流分配）
    tasks = []
    gpu_cycle = cycle(range(num_gpus))
    for name in common_files:
        gpu_id = next(gpu_cycle)
        tasks.append((name, ref_files[name], cmp_files[name], gpu_id))
        
    # 使用多进程池并行处理，每个进程绑定一个 GPU
    with Pool(num_gpus) as pool:
        results = pool.map(process_video, tasks)
    
    # 汇总所有指标
    all_metrics = {'PSNR': [], 'SSIM': [], 'LPIPS': [], 'FID': []}
    for res in results:
        all_metrics['PSNR'].append(res['PSNR'])
        all_metrics['SSIM'].append(res['SSIM'])
        all_metrics['LPIPS'].append(res['LPIPS'])
        all_metrics['FID'].append(res['FID'])
        
    final_results = {k: np.mean(v) for k, v in all_metrics.items()}
    print("\nFinal average results:")
    print(f"PSNR: {final_results['PSNR']:.2f} dB")
    print(f"SSIM: {final_results['SSIM']:.4f}")
    print(f"LPIPS: {final_results['LPIPS']:.4f}")
    print(f"FID: {final_results['FID']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_dir', type=str, default="/home/xuran/ProfilingDiT/ori/HunyuanVideo/no_cache",
                        help='参考视频目录')
    parser.add_argument('--cmp_dir', type=str, default="/home/xuran/ProfilingDiT/HunyuanVideo/results_step_inverse",
                        help='对比视频目录')
    parser.add_argument('--num_gpus', type=int, default=8, help='可用 GPU 数量')
    args = parser.parse_args()
    
    main(args.ref_dir, args.cmp_dir, args.num_gpus)
