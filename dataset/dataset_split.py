import os
import random
import shutil
from pathlib import Path


def split_dataset(src_dir, dst_dir, train_ratio=0.8):
    """
    将 AID 数据集按给定比例分割为训练集和测试集，并复制到新的目录结构中。

    """

    # 确保目标根目录存在
    os.makedirs(dst_dir, exist_ok=True)

    # 创建 test 和 train 文件夹
    # train_dir = os.path.join(dst_dir, 'train')
    # test_dir = os.path.join(dst_dir, 'test')
    # os.makedirs(train_dir, exist_ok=True)
    # os.makedirs(test_dir, exist_ok=True)

    # 获取源目录下所有类别（子文件夹）
    classes = os.listdir(src_dir)

    # 遍历每个类别
    for class_name in classes:
        class_path = os.path.join(src_dir, class_name)

        # 确保类别文件夹是目录
        if os.path.isdir(class_path):
            # 创建 test 和 train_0.1 中的子类别文件夹
            os.makedirs(os.path.join(dst_dir, class_name), exist_ok=True)
            # os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

            # 获取当前类别下所有图像文件
            images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

            # 打乱图像文件列表
            random.shuffle(images)

            # 根据比例分割图像
            split_index = int(len(images) * train_ratio)
            train_images = images[:split_index]
            test_images = images[split_index:]

            # 将训练集图像复制到 test 文件夹
            # for img in train_images:
            #     src_img_path = os.path.join(class_path, img)
            #     dst_img_path = os.path.join(train_dir, class_name, img)
            #     shutil.copy(src_img_path, dst_img_path)

            # 将测试集图像复制到 train_0.1 文件夹
            for img in train_images:
                src_img_path = os.path.join(class_path, img)
                dst_img_path = os.path.join(dst_dir, class_name, img)
                shutil.copy(src_img_path, dst_img_path)

   # print(f"Data split complete. Training data saved in {train_dir}, Testing data saved in {test_dir}.")


# 使用示例：
src_dir = '../../../../Desktop/RSIC_ResNet/dataset/EuroSAT_RGB_split/train'  # AID 数据集原始路径
dst_dir = '../../../../Desktop/RSIC_ResNet/dataset/EuroSAT_RGB_split/train_linear50'  # 分割后存放训练和测试数据的根目录

split_dataset(src_dir, dst_dir, train_ratio=0.71)
