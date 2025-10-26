import os
import cv2
import numpy as np
from PIL import Image
import random
import shutil
from typing import Tuple, List
import logging
from .utils import settings

logger = logging.getLogger(__name__)

class TextureDataProcessor:
    """纹理数据处理器"""
    
    def __init__(self):
        self.raw_data_path = settings.raw_data_path
        self.processed_data_path = settings.processed_data_path
        self.texture_classes = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
        
    def add_salt_pepper_noise(self, image: np.ndarray, salt_prob: float = 0.01, pepper_prob: float = 0.01) -> np.ndarray:
        """
        添加椒盐噪声
        """
        noisy_image = np.copy(image)
        
        # 添加盐噪声 (白点)
        salt_mask = np.random.random(image.shape) < salt_prob
        noisy_image[salt_mask] = 255
        
        # 添加椒噪声 (黑点)
        pepper_mask = np.random.random(image.shape) < pepper_prob
        noisy_image[pepper_mask] = 0
        
        return noisy_image
    
    def process_single_image(self, image_path: str, output_path: str, add_noise: bool = False) -> bool:
        """处理单张图像"""
        try:
            # 读取图像
            image = Image.open(image_path)
            
            # 转换为灰度图
            if image.mode != 'L':
                image = image.convert('L')
            image_array = np.array(image)
            
            # 添加噪声（仅对测试集）
            if add_noise:
                image_array = self.add_salt_pepper_noise(image_array, salt_prob=0.002, pepper_prob=0.002)
                logger.debug(f"为图像添加噪声: {os.path.basename(output_path)}")
            
            # 保存处理后的图像
            cv2.imwrite(output_path, image_array)
            return True
            
        except Exception as e:
            logger.error(f"处理图像 {image_path} 时出错: {e}")
            return False
    
    def process_texture_class(self, class_name: str) -> Tuple[int, int]:
        """处理单个纹理类别"""
        class_dir = os.path.join(self.raw_data_path, class_name)
        train_dir = os.path.join(self.processed_data_path, f"{class_name}_train")
        test_dir = os.path.join(self.processed_data_path, f"{class_name}_test")
        
        # 创建输出目录
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        if not os.path.exists(class_dir):
            logger.warning(f"找不到类别目录: {class_dir}")
            return 0, 0
        
        # 获取图像文件
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))]
        
        if len(image_files) < 16:
            logger.warning(f"类别 {class_name} 图像数量不足: {len(image_files)}/16")
            return 0, 0
        
        # 随机划分训练测试集
        random.shuffle(image_files)
        train_files = image_files[:10]
        test_files = image_files[10:16]
        
        # 处理训练图像
        train_count = 0
        for i, filename in enumerate(train_files):
            input_path = os.path.join(class_dir, filename)
            output_path = os.path.join(train_dir, f"{class_name}_{i+1}.png")
            if self.process_single_image(input_path, output_path, add_noise=False):
                train_count += 1
        
        # 处理测试图像
        test_count = 0
        for i, filename in enumerate(test_files):
            input_path = os.path.join(class_dir, filename)
            output_path = os.path.join(test_dir, f"{class_name}_{i+1}.png")
            if self.process_single_image(input_path, output_path, add_noise=True):
                test_count += 1
        
        logger.info(f"类别 {class_name}: 处理了 {train_count} 训练 + {test_count} 测试图像")
        return train_count, test_count
    
    def process_all_data(self) -> bool:
        """处理所有数据"""
        logger.info("开始处理纹理数据...")
        
        # 清空处理目录
        if os.path.exists(self.processed_data_path):
            shutil.rmtree(self.processed_data_path)
        os.makedirs(self.processed_data_path)
        
        total_train = 0
        total_test = 0
        
        # 处理每个类别
        for class_name in self.texture_classes:
            train_count, test_count = self.process_texture_class(class_name)
            total_train += train_count
            total_test += test_count
        
        logger.info(f"数据处理完成! 总计: {total_train} 训练 + {total_test} 测试图像")
        return total_train > 0 and total_test > 0
    
    def verify_data_structure(self) -> bool:
        """验证数据目录结构"""
        logger.info("验证数据目录结构...")
        
        if not os.path.exists(self.processed_data_path):
            logger.error("处理数据目录不存在")
            return False
        
        valid = True
        for class_name in self.texture_classes:
            train_dir = os.path.join(self.processed_data_path, f"{class_name}_train")
            test_dir = os.path.join(self.processed_data_path, f"{class_name}_test")
            
            train_exists = os.path.exists(train_dir)
            test_exists = os.path.exists(test_dir)
            
            if not train_exists or not test_exists:
                logger.error(f"类别 {class_name} 目录不完整")
                valid = False
                continue
            
            train_files = len([f for f in os.listdir(train_dir) if f.endswith('.png')])
            test_files = len([f for f in os.listdir(test_dir) if f.endswith('.png')])
            
            logger.info(f"{class_name}: 训练 {train_files} 张, 测试 {test_files} 张")
        
        return valid