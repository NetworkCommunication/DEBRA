import os
import numpy as np
from PIL import Image

def calculate_mean_and_variance(dataset_dir):
    mean = np.array([0.0, 0.0, 0.0])
    variance = np.array([0.0, 0.0, 0.0])
    total_images = 0

    # 遍历目录，找到所有图片
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', 'tif')):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path)
                    img = img.convert('RGB')  # 确保是RGB模式
                    img_array = np.array(img, dtype=np.float32)
                    img_array /= 255.0  # 归一化到 [0, 1]

                    # 检查图像数组的形状
                    if img_array.ndim == 3:  # 确保是三维数组
                        # 计算当前批次均值和方差
                        batch_mean = np.mean(img_array, axis=(0, 1))
                        batch_variance = np.var(img_array, axis=(0, 1))

                        # 更新总体均值和方差
                        total_images += 1
                        mean += (batch_mean - mean) / total_images
                        variance += (batch_variance - variance) / total_images
                    else:
                        print(f"Skipped image due to unexpected shape: {img_path} with shape {img_array.shape}")

                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    continue

    return mean, variance

dataset_directory = 'UCMerced'
mean, variance = calculate_mean_and_variance(dataset_directory)
print("Mean:", mean)
print("Variance:", variance)
