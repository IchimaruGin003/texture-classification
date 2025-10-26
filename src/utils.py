import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Settings(BaseModel):
    """应用配置"""
    # 基础配置
    app_name: str = Field(..., alias="APP_NAME")
    debug: bool = Field(False, alias="DEBUG")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    
    # 路径配置
    raw_data_path: str = Field(..., alias="RAW_DATA_PATH")
    processed_data_path: str = Field(..., alias="PROCESSED_DATA_PATH")
    model_save_path: str = Field(..., alias="MODEL_SAVE_PATH")
    
    # MLflow配置
    mlflow_tracking_uri: str = Field(..., alias="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(..., alias="MLFLOW_EXPERIMENT_NAME")
    
    # 模型参数
    knn_neighbors: int = Field(..., alias="KNN_NEIGHBORS")
    test_size: float = Field(..., alias="TEST_SIZE")
    random_state: int = Field(..., alias="RANDOM_STATE")
    
    # DagsHub配置（可选）
    dagshub_username: Optional[str] = Field(None, alias="DAGSHUB_USERNAME")
    dagshub_repo_name: Optional[str] = Field(None, alias="DAGSHUB_REPO_NAME")

def get_settings() -> Settings:
    """获取配置实例"""
    return Settings(**os.environ)

# 全局配置实例
settings = get_settings()