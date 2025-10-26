import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import logging
from typing import Tuple, Dict, Any
import joblib
from .utils import settings

logger = logging.getLogger(__name__)

class TextureModelTrainer:
    """纹理分类模型训练器"""
    
    def __init__(self):
        self.knn_neighbors = settings.knn_neighbors
        self.random_state = settings.random_state
        self.model_save_path = settings.model_save_path
        
        # 确保模型保存目录存在
        os.makedirs(self.model_save_path, exist_ok=True)
    
    def prepare_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """准备训练和测试数据"""
        # 分离训练集和测试集
        train_df = features_df[features_df['set_type'] == 'train']
        test_df = features_df[features_df['set_type'] == 'test']
        
        # 特征列
        feature_cols = ['energy', 'contrast', 'correlation', 'entropy']
        
        X_train = train_df[feature_cols].values
        y_train = train_df['class'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['class'].values
        
        logger.info(f"数据准备完成: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[KNeighborsClassifier, StandardScaler]:
        """训练KNN模型"""
        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # 训练KNN分类器
        knn = KNeighborsClassifier(n_neighbors=self.knn_neighbors)
        knn.fit(X_train_scaled, y_train)
        
        logger.info(f"模型训练完成: KNN (k={self.knn_neighbors})")
        return knn, scaler
    
    def evaluate_model(self, model: KNeighborsClassifier, scaler: StandardScaler, 
                      X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """评估模型性能"""
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info(f"模型评估完成: 准确率 {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred
        }
    
    def feature_importance_analysis(self, model: KNeighborsClassifier, scaler: StandardScaler,
                                  X_test: np.ndarray, y_test: np.ndarray, feature_names: list) -> pd.DataFrame:
        """特征重要性分析"""
        X_test_scaled = scaler.transform(X_test)
        
        # 计算置换重要性
        perm_importance = permutation_importance(
            model, X_test_scaled, y_test,
            n_repeats=10,
            random_state=self.random_state
        )
        
        # 整理结果
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': perm_importance.importances_mean
        }).sort_values('importance', ascending=False)
        
        logger.info("特征重要性分析完成")
        return importance_df
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, save_path: str = None):
        """绘制特征重要性图"""
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('特征重要性得分')
        plt.title('GLCM特征对纹理分类的重要性')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"特征重要性图保存到: {save_path}")
        
        plt.close()
    
    def save_model(self, model: KNeighborsClassifier, scaler: StandardScaler, accuracy: float):
        """保存模型和标准化器"""
        # 保存模型
        model_path = os.path.join(self.model_save_path, f"knn_model_{accuracy:.4f}.pkl")
        scaler_path = os.path.join(self.model_save_path, "scaler.pkl")
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"模型保存到: {model_path}")
        logger.info(f"标准化器保存到: {scaler_path}")
        
        return model_path, scaler_path
    
    def train_and_evaluate_with_mlflow(self, features_df: pd.DataFrame, run_name: str = None):
        """使用MLflow进行训练和评估"""
        # 设置MLflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        
        with mlflow.start_run(run_name=run_name):
            # 记录参数
            mlflow.log_param("knn_neighbors", self.knn_neighbors)
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_param("feature_count", 4)  # 4个GLCM特征
            mlflow.log_param("dataset", "Brodatz_Texture")
            
            # 准备数据
            X_train, X_test, y_train, y_test = self.prepare_data(features_df)
            
            # 训练模型
            model, scaler = self.train_model(X_train, y_train)
            
            # 评估模型
            evaluation = self.evaluate_model(model, scaler, X_test, y_test)
            accuracy = evaluation['accuracy']
            
            # 记录指标
            mlflow.log_metric("accuracy", accuracy)
            
            # 特征重要性分析
            feature_names = ['energy', 'contrast', 'correlation', 'entropy']
            importance_df = self.feature_importance_analysis(model, scaler, X_test, y_test, feature_names)
            
            # 记录特征重要性
            for _, row in importance_df.iterrows():
                mlflow.log_metric(f"importance_{row['feature']}", row['importance'])
            
            # 保存特征重要性图
            importance_plot_path = os.path.join(self.model_save_path, "feature_importance.png")
            self.plot_feature_importance(importance_df, importance_plot_path)
            mlflow.log_artifact(importance_plot_path)
            
            # 保存模型
            model_path, scaler_path = self.save_model(model, scaler, accuracy)
            
            # 记录模型
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(scaler_path)
            
            # 记录混淆矩阵
            cm_path = os.path.join(self.model_save_path, "confusion_matrix.png")
            self.plot_confusion_matrix(evaluation['confusion_matrix'], 
                                     classes=np.unique(y_test), 
                                     save_path=cm_path)
            mlflow.log_artifact(cm_path)
            
            logger.info(f"MLflow实验记录完成，准确率: {accuracy:.4f}")
            
            return {
                'model': model,
                'scaler': scaler,
                'accuracy': accuracy,
                'feature_importance': importance_df,
                'evaluation': evaluation,
                'model_path': model_path,
                'scaler_path': scaler_path
            }
    
    def plot_confusion_matrix(self, cm: np.ndarray, classes: list, save_path: str = None):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('混淆矩阵')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # 在矩阵中显示数值
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        plt.close()