import pytest
import numpy as np
from src.feature_extractor import GLCMFeatureExtractor

class TestGLCMFeatureExtractor:
    """测试GLCM特征提取器"""
    
    def setup_method(self):
        """测试设置"""
        self.extractor = GLCMFeatureExtractor(distance=1)
    
    def test_manual_glcm(self):
        """测试手动计算GLCM"""
        # 创建简单的测试图像
        test_image = np.array([
            [1, 1, 2],
            [1, 2, 2],
            [2, 2, 3]
        ], dtype=np.uint8)
        
        glcm = self.extractor.manual_glcm(test_image, angle=0)
        
        # 检查GLCM的基本属性
        assert glcm.shape == (256, 256)
        assert np.sum(glcm) > 0  # 应该非零
    
    def test_calculate_contrast(self):
        """测试计算对比度"""
        # 创建简单的GLCM矩阵
        test_glcm = np.zeros((4, 4))
        test_glcm[1, 3] = 0.5  # 高对比度
        test_glcm[1, 1] = 0.5  # 零对比度
        
        contrast = self.extractor.calculate_contrast(test_glcm)
        
        # 对比度应该是 (1-3)^2 * 0.5 = 4 * 0.5 = 2
        expected_contrast = (1-3)**2 * 0.5
        assert contrast == pytest.approx(expected_contrast)
    
    def test_calculate_energy(self):
        """测试计算能量"""
        test_glcm = np.array([
            [0.25, 0.25],
            [0.25, 0.25]
        ])
        
        energy = self.extractor.calculate_energy(test_glcm)
        expected_energy = 0.25  # 0.25^2 * 4 = 0.25
        assert energy == pytest.approx(expected_energy)
    
    def test_extract_features(self):
        """测试特征提取"""
        # 创建测试图像
        test_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        features = self.extractor.extract_features(test_image)
        
        # 检查返回的特征
        expected_features = ['energy', 'contrast', 'correlation', 'entropy']
        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], float)