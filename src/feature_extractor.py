import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import logging
from typing import Dict, List, Optional
from .utils import settings

logger = logging.getLogger(__name__)

class GLCMFeatureExtractor:
    """GLCM特征提取器"""
    
    def __init__(self, distance: int = 1):
        self.distance = distance
        self.angles = [0, 45, 90, 135]  # 0°, 45°, 90°, 135°
    
    def manual_glcm(self, image: np.ndarray, angle: int) -> np.ndarray:
        """手动计算灰度共生矩阵"""
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # 角度转换为偏移量
        if angle == 0:    dx, dy = self.distance, 0
        elif angle == 45: dx, dy = self.distance, self.distance
        elif angle == 90: dx, dy = 0, self.distance
        elif angle == 135: dx, dy = -self.distance, self.distance
        else: 
            raise ValueError("角度必须是0, 45, 90, 135度")
        
        glcm = np.zeros((256, 256), dtype=np.float64)
        height, width = image.shape
        
        for i in range(height):
            for j in range(width):
                i2, j2 = i + dy, j + dx
                if 0 <= i2 < height and 0 <= j2 < width:
                    gray1, gray2 = image[i, j], image[i2, j2]
                    glcm[gray1, gray2] += 1
        
        if np.sum(glcm) > 0:
            glcm /= np.sum(glcm)
        
        return glcm
    
    def calculate_contrast(self, glcm: np.ndarray) -> float:
        """计算对比度"""
        return np.sum([glcm[i, j] * (i - j) ** 2 for i in range(256) for j in range(256)])
    
    def calculate_energy(self, glcm: np.ndarray) -> float:
        """计算能量"""
        return np.sum(glcm ** 2)
    
    def calculate_correlation(self, glcm: np.ndarray) -> float:
        """计算相关性"""
        i_idx, j_idx = np.indices(glcm.shape)
        mean_i, mean_j = np.sum(i_idx * glcm), np.sum(j_idx * glcm)
        std_i = np.sqrt(np.sum((i_idx - mean_i) ** 2 * glcm))
        std_j = np.sqrt(np.sum((j_idx - mean_j) ** 2 * glcm))
        return np.sum((i_idx - mean_i) * (j_idx - mean_j) * glcm) / (std_i * std_j) if std_i * std_j != 0 else 0
    
    def calculate_entropy(self, glcm: np.ndarray) -> float:
        """计算熵"""
        return -np.sum([glcm[i, j] * np.log2(glcm[i, j] + 1e-10) for i in range(256) for j in range(256) if glcm[i, j] > 0])
    
    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """提取GLCM特征"""
        features = {}
        metrics = {'energy': [], 'contrast': [], 'correlation': [], 'entropy': []}
        
        for angle in self.angles:
            glcm = self.manual_glcm(image, angle)
            metrics['energy'].append(self.calculate_energy(glcm))
            metrics['contrast'].append(self.calculate_contrast(glcm))
            metrics['correlation'].append(self.calculate_correlation(glcm))
            metrics['entropy'].append(self.calculate_entropy(glcm))
        
        for key in metrics:
            features[key] = np.mean(metrics[key])
        
        return features
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """加载图像"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                image = np.array(Image.open(image_path).convert('L'))
            return image
        except Exception as e:
            logger.error(f"无法读取图像 {image_path}: {e}")
            return None
    
    def extract_features_from_directory(self, data_path: str) -> pd.DataFrame:
        """从目录提取所有图像特征"""
        features_list = []
        
        for class_num in range(1, 7):
            class_name = f"D{class_num}"
            
            for set_type in ['train', 'test']:
                img_dir = os.path.join(data_path, f"{class_name}_{set_type}")
                
                if os.path.exists(img_dir):
                    for filename in os.listdir(img_dir):
                        if filename.endswith('.png'):
                            image_path = os.path.join(img_dir, filename)
                            image = self.load_image(image_path)
                            
                            if image is not None:
                                try:
                                    features = self.extract_features(image)
                                    features['filename'] = filename
                                    features['class'] = class_name
                                    features['set_type'] = set_type
                                    features_list.append(features)
                                    logger.debug(f"处理: {class_name}_{set_type}/{filename}")
                                except Exception as e:
                                    logger.error(f"处理 {filename} 时出错: {e}")
        
        return pd.DataFrame(features_list)