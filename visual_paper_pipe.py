import cv2
import numpy as np
import os

def add_gaussian_noise(image, sigma):
    """向图像添加高斯噪声"""
    noise = np.random.normal(0, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)  # 保证结果仍然在 0-255 范围内
    return noisy_image

def process_video(video_path, output_folder, noise_levels):
    """从视频提取第一帧，增加不同程度的高斯噪声，并保存"""
    os.makedirs(output_folder, exist_ok=True)

    # 读取视频
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("无法读取视频")
        return
    
    # 保存原始第一帧
    cv2.imwrite(os.path.join(output_folder, "frame_0_original.jpg"), frame)

    # 逐渐增加噪声并保存图像（包括低噪声和高噪声）
    for i, sigma in enumerate(noise_levels):
        noisy_frame = add_gaussian_noise(frame, sigma)
        cv2.imwrite(os.path.join(output_folder, f"frame_{i+1}_noise{sigma}.jpg"), noisy_frame)

    # 生成低噪声版本
    low_noise_levels = [1, 3, 5]  
    for i, sigma in enumerate(low_noise_levels):
        noisy_frame = add_gaussian_noise(frame, sigma)
        cv2.imwrite(os.path.join(output_folder, f"frame_low_{i+1}_noise{sigma}.jpg"), noisy_frame)

    # 生成完全高斯噪声的图片
    gaussian_noise_only = np.random.normal(127, 50, frame.shape).astype(np.uint8)  # 以 127 为中心
    cv2.imwrite(os.path.join(output_folder, "frame_full_gaussian_noise.jpg"), gaussian_noise_only)

    print(f"处理完成，图像保存在 {output_folder}")

# 使用示例
video_path = "/home/xuran/DIT-Cache/HunyuanVideo/results/2025-02-23-16:22:03_seed42_A cat walks on the grass, realistic style..mp4"  # 替换为你的视频路径
output_folder = "output_frames_pipe"
noise_levels = [10, 20, 30, 50, 70, 100]  # 原始噪声级别

process_video(video_path, output_folder, noise_levels)
