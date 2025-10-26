import pytest
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from src.data_processor import TextureDataProcessor
from src.utils import settings

class TestTextureDataProcessor:
    """测试纹理数据处理器"""
    
    def setup_method(self):
        """测试设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.temp_dir, "raw")
        self.processed_dir = os.path.join(self.temp_dir, "processed")
        
        os.makedirs(self.raw_dir, exist_ok=True)
        
        # 创建测试图像目录结构
        for class_name in ['D1', 'D2']:
            class_dir = os.path.join(self.raw_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # 创建一些测试图像文件
            for i in range(16):
                test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
                image_path = os.path.join(class_dir, f"test_{i}.tif")
                from PIL import Image
                Image.fromarray(test_image).save(image_path)
    
    def test_add_salt_pepper_noise(self):
        """测试添加椒盐噪声"""
        processor = TextureDataProcessor()
        
        # 创建测试图像
        test_image = np.ones((10, 10), dtype=np.uint8) * 128
        
        # 添加噪声
        noisy_image = processor.add_salt_pepper_noise(test_image, salt_prob=0.1, pepper_prob=0.1)
        
        # 检查噪声添加
        assert noisy_image.shape == test_image.shape
        assert np.any(noisy_image == 0) or np.any(noisy_image == 255)  # 应该有噪声点
    
    def test_process_single_image(self):
        """测试处理单张图像"""
        processor = TextureDataProcessor()
        
        # 创建测试图像
        test_image_path = os.path.join(self.temp_dir, "test_input.tif")
        test_output_path = os.path.join(self.temp_dir, "test_output.png")
        
        test_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        from PIL import Image
        Image.fromarray(test_image).save(test_image_path)
        
        # 测试处理（无噪声）
        result = processor.process_single_image(test_image_path, test_output_path, add_noise=False)
        assert result == True
        assert os.path.exists(test_output_path)
    
    @patch('src.data_processor.settings')
    def test_initialization(self, mock_settings):
        """测试初始化"""
        mock_settings.raw_data_path = "/test/raw"
        mock_settings.processed_data_path = "/test/processed"
        
        processor = TextureDataProcessor()
        assert processor.raw_data_path == "/test/raw"
        assert processor.processed_data_path == "/test/processed"
        assert len(processor.texture_classes) == 6
    
    def teardown_method(self):
        """测试清理"""
        import shutil
        shutil.rmtree(self.temp_dir)