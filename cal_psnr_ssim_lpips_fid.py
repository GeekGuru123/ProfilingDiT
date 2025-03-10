import os
import cv2
import numpy as np
from sewar import psnr, ssim
import lpips
import torch
from pytorch_fid import fid_score
from tqdm import tqdm

# 初始化LPIPS模型
loss_fn_alex = lpips.LPIPS(net='alex').cuda()

def read_video_frames(video_path):
    """读取视频帧并返回numpy数组"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames)

def calculate_metrics(vid1, vid2):
    """计算两个视频序列之间的指标"""
    psnr_values = []
    ssim_values = []
    lpips_values = []
    
    # 确保视频长度相同
    min_len = min(len(vid1), len(vid2))
    vid1 = vid1[:min_len]
    vid2 = vid2[:min_len]
    
    # 预处理FID需要的图像文件
    os.makedirs('temp/ref', exist_ok=True)
    os.makedirs('temp/gen', exist_ok=True)
    
    for i in tqdm(range(min_len)):
        # PSNR和SSIM计算
        psnr_val = psnr(vid1[i], vid2[i])
        ssim_val, _ = ssim(vid1[i], vid2[i])
        
        # LPIPS计算
        img1 = lpips.im2tensor(vid1[i]).cuda()
        img2 = lpips.im2tensor(vid2[i]).cuda()
        lpips_val = loss_fn_alex(img1, img2).item()
        
        # 保存结果
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        lpips_values.append(lpips_val)
        
        # 保存FID需要的图像
        cv2.imwrite(f'temp/ref/{i:05d}.png', cv2.cvtColor(vid1[i], cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'temp/gen/{i:05d}.png', cv2.cvtColor(vid2[i], cv2.COLOR_RGB2BGR))
    
    # 计算FID
    fid = fid_score.calculate_fid_given_paths(['temp/ref', 'temp/gen'], 256, torch.device('cuda'), 2048)
    
    return {
        'PSNR': np.mean(psnr_values),
        'SSIM': np.mean(ssim_values),
        'LPIPS': np.mean(lpips_values),
        'FID': fid
    }

def main(ref_dir, cmp_dir):
    # 获取视频列表
    ref_files = {f.split('.')[0]: os.path.join(ref_dir, f) 
                 for f in os.listdir(ref_dir) if f.endswith(('.mp4', '.avi', '.mov'))}
    cmp_files = {f.split('.')[0]: os.path.join(cmp_dir, f) 
                 for f in os.listdir(cmp_dir) if f.endswith(('.mp4', '.avi', '.mov'))}
    
    common_files = set(ref_files.keys()) & set(cmp_files.keys())
    print(f"找到 {len(common_files)} 个共同视频文件")
    
    # 收集所有指标
    all_metrics = {'PSNR': [], 'SSIM': [], 'LPIPS': [], 'FID': []}
    
    for name in common_files:
        print(f"\n处理视频: {name}")
        ref_vid = read_video_frames(ref_files[name])
        cmp_vid = read_video_frames(cmp_files[name])
        
        metrics = calculate_metrics(ref_vid, cmp_vid)
        
        for k in metrics:
            all_metrics[k].append(metrics[k])
        
        print(f"当前结果: PSNR={metrics['PSNR']:.2f}, SSIM={metrics['SSIM']:.3f}, "
              f"LPIPS={metrics['LPIPS']:.3f}, FID={metrics['FID']:.1f}")
    
    # 计算平均值
    final_results = {k: np.mean(v) for k, v in all_metrics.items()}
    print("\n最终平均结果:")
    print(f"PSNR: {final_results['PSNR']:.2f} dB")
    print(f"SSIM: {final_results['SSIM']:.4f}")
    print(f"LPIPS: {final_results['LPIPS']:.4f}")
    print(f"FID: {final_results['FID']:.2f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_dir', type=str, default="/home/xuran/ProfilingDiT/ori/HunyuanVideo/prompt_final", help='参考视频目录')
    parser.add_argument('--cmp_dir', type=str, default="/home/xuran/ProfilingDiT/HunyuanVideo/results_cache", help='对比视频目录')
    args = parser.parse_args()
    
    main(args.ref_dir, args.cmp_dir)