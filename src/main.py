#!/usr/bin/env python3
"""
纹理分类主应用程序
"""

import os
import sys
import logging
import argparse
from dotenv import load_dotenv

# 添加src到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processor import TextureDataProcessor
from src.feature_extractor import GLCMFeatureExtractor
from src.model_trainer import TextureModelTrainer
from src.utils import settings, get_settings

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def setup_directories():
    """设置必要的目录结构"""
    directories = [
        settings.raw_data_path,
        settings.processed_data_path,
        settings.model_save_path,
        os.path.dirname(settings.mlflow_tracking_uri)
    ]
    
    for directory in directories:
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"创建目录: {directory}")

def process_data():
    """数据处理流程"""
    logger.info("开始数据处理...")
    processor = TextureDataProcessor()
    
    # 处理数据
    success = processor.process_all_data()
    if not success:
        logger.error("数据处理失败")
        return False
    
    # 验证数据
    if not processor.verify_data_structure():
        logger.error("数据验证失败")
        return False
    
    logger.info("数据处理完成")
    return True

def extract_features():
    """特征提取流程"""
    logger.info("开始特征提取...")
    extractor = GLCMFeatureExtractor()
    
    # 提取特征
    features_df = extractor.extract_features_from_directory(settings.processed_data_path)
    
    if len(features_df) == 0:
        logger.error("没有提取到任何特征")
        return None
    
    # 保存特征到CSV
    features_csv_path = os.path.join(settings.processed_data_path, "glcm_features.csv")
    features_df.to_csv(features_csv_path, index=False)
    logger.info(f"特征保存到: {features_csv_path}")
    
    return features_df

def train_model(features_df, run_name="training_run"):
    """模型训练流程"""
    logger.info("开始模型训练...")
    trainer = TextureModelTrainer()
    
    # 使用MLflow进行训练和评估
    results = trainer.train_and_evaluate_with_mlflow(features_df, run_name=run_name)
    
    logger.info(f"模型训练完成，准确率: {results['accuracy']:.4f}")
    return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="纹理分类应用程序")
    parser.add_argument('--process-data', action='store_true', help='处理原始数据')
    parser.add_argument('--extract-features', action='store_true', help='提取GLCM特征')
    parser.add_argument('--train-model', action='store_true', help='训练模型')
    parser.add_argument('--run-name', type=str, default='texture_classification', help='MLflow运行名称')
    parser.add_argument('--all', action='store_true', help='运行完整流程')
    
    args = parser.parse_args()
    
    # 加载环境变量
    load_dotenv()
    
    # 设置目录
    setup_directories()
    
    logger.info(f"启动纹理分类应用: {settings.app_name}")
    
    # 根据参数执行相应流程
    if args.all or args.process_data:
        if not process_data():
            return
    
    features_df = None
    if args.all or args.extract_features:
        features_df = extract_features()
        if features_df is None:
            return
    
    if args.all or args.train_model:
        if features_df is None:
            # 尝试从文件加载特征
            features_csv_path = os.path.join(settings.processed_data_path, "glcm_features.csv")
            if os.path.exists(features_csv_path):
                import pandas as pd
                features_df = pd.read_csv(features_csv_path)
                logger.info(f"从文件加载特征: {features_csv_path}")
            else:
                logger.error("没有可用的特征数据，请先运行特征提取")
                return
        
        train_model(features_df, args.run_name)
    
    if not any([args.process_data, args.extract_features, args.train_model, args.all]):
        parser.print_help()

if __name__ == "__main__":
    main()