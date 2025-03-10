import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 生成时间步
num_steps = 300
x = np.linspace(0, 1, num_steps)

# 速度曲线：先快后慢 (指数衰减)
progress = 1 - np.exp(-4 * x)

# 生成渐变色
gradient = np.linspace(0, 1, num_steps).reshape(1, -1)

# 创建图像
fig, ax = plt.subplots(figsize=(4, 0.5))  # 仅保留进度条部分
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")  # 隐藏坐标轴

# 绘制渐变色条
ax.imshow(gradient, extent=[0, 1, 0, 1], cmap="Blues", aspect="auto")

# 绘制进度条
progress_width = progress[-1]  # 最终进度
rect = patches.Rectangle((0, 0), progress_width, 1, color="royalblue")
ax.add_patch(rect)

# 保存为 PNG
plt.savefig("progress_bar.png", dpi=300, bbox_inches="tight", pad_inches=0)
plt.show()
