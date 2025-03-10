import os
from PIL import Image, ImageDraw, ImageFont

# 定义根目录和步骤列表
root_dir = '/home/xuran/DIT-Cache/HunyuanVideo/output_images_tens_360p'
steps = [0, 10, 20, 30, 40, 49]  # 需要遍历的 step 目录
# frame_indices = [2, 15, 21, 36, 38]  # 选取的 frames
frame_indices = [12,21,37]

# 字体设置（需要系统支持的字体文件路径）
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
except:
    font = ImageFont.load_default()

# 存储所有选定的图片路径
selected_images = []
image_labels = []  # 存储每张图片对应的 Step 和 Block 信息

for step in steps:
    step_dir = os.path.join(root_dir, f'step{step}')
    if not os.path.exists(step_dir):
        print(f'警告：目录 {step_dir} 不存在，跳过。')
        continue

    for idx in frame_indices:
        image_path = os.path.join(step_dir, f'frame_0007_{idx}_attention.png')
        if os.path.exists(image_path):
            selected_images.append(image_path)
            image_labels.append(f"Step {step}, Block {idx}")
        else:
            print(f'警告：文件 {image_path} 不存在，跳过。')

# 如果没有找到任何图片，直接退出
if not selected_images:
    print("未找到符合条件的图片，程序终止。")
    exit()

# 打开所有选定的图片
opened_images = [Image.open(img_path) for img_path in selected_images]

# 假设所有图片尺寸相同，获取单张图片的尺寸
img_width, img_height = opened_images[0].size

# 计算网格的行数和列数
grid_cols = len(frame_indices)  # 每行排列的图片数（frame数量）
grid_rows = len(steps)  # 总共多少行（step数量）

# 创建一个新的空白图像（RGB模式），用于拼接
grid_image = Image.new('RGB', (grid_cols * img_width, grid_rows * img_height), (255, 255, 255))

# 将每张图片粘贴到网格的相应位置，并标注 Step 和 Block 号
for index, (img, label) in enumerate(zip(opened_images, image_labels)):
    row = index // grid_cols
    col = index % grid_cols
    grid_image.paste(img, (col * img_width, row * img_height))

    # 在图片上绘制 Step 和 Block 信息
    # draw = ImageDraw.Draw(grid_image)
    # text_position = (col * img_width + 10, row * img_height + 10)  # 文字位置
    # draw.text(text_position, label, fill=(255, 0, 0), font=font)  # 红色文字

# 保存最终的网格大图
output_path = '/home/xuran/DIT-Cache/HunyuanVideo/output_images_tens_360p/batch_grid_image1.png'
grid_image.save(output_path)
grid_image.show()

print(f'拼接完成，最终大图已保存至 {output_path}')
