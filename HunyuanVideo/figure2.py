import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置三个视频文件夹路径
PATHS = [
    "/home/xuran/ProfilingDiT/HunyuanVideo/no_cache",
    "/home/xuran/ProfilingDiT/HunyuanVideo/results_cache",
    "/home/xuran/ProfilingDiT/ori/HunyuanVideo/teacache_slow",
]  # 需要修改成你的路径

# 指定要处理的视频和帧索引
SELECTED_FRAMES = {  
    "seed42_A red sports car speeding down a winding mountain road..mp4": [10, 50, 100]
}
    # "seed42_A group of friends roasting marshmallows around a campfire..mp4": [10, 50, 100], 
    # "seed42_A chef preparing sushi in a busy kitchen..mp4": [10, 50, 100]
SAVE_DIR = "./saved_frames_19"  # 修改为你想保存的路径
os.makedirs(SAVE_DIR, exist_ok=True)

# 读取指定帧
def extract_specific_frames(video_path, frame_indices):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    for idx in frame_indices:
        if idx >= total_frames:  # 如果索引超出范围，则跳过
            print(f"Warning: {video_path} 没有足够的帧数 ({idx} >= {total_frames})")
            continue
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = np.zeros((256, 256, 3), dtype=np.uint8)  # 失败填充黑色
        frames.append((frame, idx))  # 返回帧和索引
    cap.release()
    
    return frames

# 遍历指定的视频和方法
for video_name, frame_idxs in SELECTED_FRAMES.items():
    for col, path in enumerate(PATHS):
        video_path = os.path.join(path, video_name)
        
        if not os.path.exists(video_path):
            print(f"Warning: {video_path} 不存在，跳过！")
            continue
        
        frames = extract_specific_frames(video_path, frame_idxs)  # 按指定帧数提取
        
        for frame, frame_idx in frames:
            # 保存图片
            save_path = os.path.join(SAVE_DIR, f"method{col+1}_{video_name}_frame{frame_idx}.png")
            cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"Saved: {save_path}")

print("所有指定帧保存完成。")
