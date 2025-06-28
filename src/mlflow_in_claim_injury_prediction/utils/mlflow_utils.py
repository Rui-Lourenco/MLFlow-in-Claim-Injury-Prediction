"""
MLflow utilities for the claim injury prediction project.
"""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
import logging
import os
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional

log = logging.getLogger(__name__)

def log_dataset_info(data: pd.DataFrame, dataset_name: str, description: str = "") -> None:
    """
    Log dataset information to MLflow.
    
    Args:
        data: DataFrame to log
        dataset_name: Name of the dataset (e.g., "train", "validation", "test")
        description: Description of the dataset
    """
    try:
        mlflow.log_param(f"{dataset_name}_shape", str(data.shape))
        mlflow.log_param(f"{dataset_name}_columns", len(data.columns))
        mlflow.log_param(f"{dataset_name}_rows", len(data))
        
        if len(data.select_dtypes(include=[np.number]).columns) > 0:
            numeric_stats = data.describe()
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                numeric_stats.to_csv(tmp_file.name)
                tmp_file_path = tmp_file.name
            
            mlflow.log_artifact(tmp_file_path, f"{dataset_name}_statistics.csv")
            
            try:
                os.unlink(tmp_file_path)
            except PermissionError:
                pass
        
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            missing_df = pd.DataFrame({
                'column': missing_values.index,
                'missing_count': missing_values.values,
                'missing_percentage': (missing_values.values / len(data)) * 100
            })
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                missing_df.to_csv(tmp_file.name, index=False)
                tmp_file_path = tmp_file.name
            
            mlflow.log_artifact(tmp_file_path, f"{dataset_name}_missing_values.csv")
            
            try:
                os.unlink(tmp_file_path)
            except PermissionError:
                pass
        
        sample_data = data.head(100)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            sample_data.to_csv(tmp_file.name, index=False)
            tmp_file_path = tmp_file.name
        
        mlflow.log_artifact(tmp_file_path, f"{dataset_name}_sample.csv")
        
        try:
            os.unlink(tmp_file_path)
        except PermissionError:
            pass
        
        mlflow.set_tag(f"{dataset_name}_description", description)
        mlflow.set_tag(f"{dataset_name}_shape", str(data.shape))
        
        log.info(f"Logged dataset info for {dataset_name}")
        
    except Exception as e:
        log.warning(f"Failed to log dataset info for {dataset_name}: {e}")

def log_model_with_metadata(model, model_name: str, model_type: str, 
                          feature_names: list, model_params: dict, input_example: pd.DataFrame = None) -> None:
    """
    Log model with comprehensive metadata.
    
    Args:
        model: Trained model object
        model_name: Name of the model
        model_type: Type of model (e.g., 'xgboost', 'random_forest')
        feature_names: List of feature names
        model_params: Model parameters
        input_example: Example input data for model signature inference
    """
    try:
        for param_name, param_value in model_params.items():
            mlflow.log_param(f"{model_name}_{param_name}", param_value)
        
        mlflow.log_param(f"{model_name}_type", model_type)
        mlflow.log_param(f"{model_name}_features_count", len(feature_names))
        mlflow.log_param(f"{model_name}_feature_names", str(feature_names))
        
        if model_type == "xgboost":
            mlflow.xgboost.log_model(model, f"{model_name}_model", input_example=input_example)
        elif model_type == "random_forest":
            mlflow.sklearn.log_model(model, f"{model_name}_model", input_example=input_example)
        else:
            mlflow.sklearn.log_model(model, f"{model_name}_model", input_example=input_example)
        
        log.info(f"Logged model {model_name} ({model_type})")
        
    except Exception as e:
        log.error(f"Failed to log model {model_name}: {e}")
        raise

def log_feature_importance(feature_importance: pd.DataFrame, model_name: str) -> None:
    """
    Log feature importance with proper formatting.
    
    Args:
        feature_importance: DataFrame with feature importance
        model_name: Name of the model
    """
    try:
        if 'feature' not in feature_importance.columns or 'importance' not in feature_importance.columns:
            log.warning(f"Feature importance DataFrame for {model_name} doesn't have expected columns")
            return
        
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        top_features = feature_importance.head(20)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            top_features.to_csv(tmp_file.name, index=False)
            tmp_file_path = tmp_file.name
        
        mlflow.log_artifact(tmp_file_path, f"{model_name}_top_features.csv")
        
        try:
            os.unlink(tmp_file_path)
        except PermissionError:
            pass
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            feature_importance.to_csv(tmp_file.name, index=False)
            tmp_file_path = tmp_file.name
        
        mlflow.log_artifact(tmp_file_path, f"{model_name}_all_features.csv")
        
        try:
            os.unlink(tmp_file_path)
        except PermissionError:
            pass
        
        importance_stats = {
            'total_features': len(feature_importance),
            'max_importance': feature_importance['importance'].max(),
            'min_importance': feature_importance['importance'].min(),
            'mean_importance': feature_importance['importance'].mean(),
            'std_importance': feature_importance['importance'].std()
        }
        
        for stat_name, stat_value in importance_stats.items():
            mlflow.log_metric(f"{model_name}_{stat_name}", stat_value)
        
        log.info(f"Logged feature importance for {model_name}")
        
    except Exception as e:
        log.error(f"Failed to log feature importance for {model_name}: {e}")

def log_predictions_with_metadata(y_true: np.ndarray, y_pred: np.ndarray, 
                                y_proba: np.ndarray, dataset_name: str) -> None:
    """
    Log predictions with comprehensive metadata.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        dataset_name: Name of the dataset
    """
    try:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        predictions_df = pd.DataFrame({
            'true_label': y_true,
            'predicted_label': y_pred,
            'prediction_confidence': np.max(y_proba, axis=1)
        })
        
        for i in range(y_proba.shape[1]):
            predictions_df[f'prob_class_{i}'] = y_proba[:, i]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            predictions_df.to_csv(tmp_file.name, index=False)
            tmp_file_path = tmp_file.name
        
        mlflow.log_artifact(tmp_file_path, f"{dataset_name}_predictions.csv")
        
        try:
            os.unlink(tmp_file_path)
        except PermissionError:
            pass
        
        mlflow.log_metric(f"{dataset_name}_total_predictions", len(y_pred))
        mlflow.log_metric(f"{dataset_name}_unique_predictions", len(np.unique(y_pred)))
        mlflow.log_metric(f"{dataset_name}_avg_confidence", predictions_df['prediction_confidence'].mean())
        
        log.info(f"Logged predictions for {dataset_name}")
        
    except Exception as e:
        log.error(f"Failed to log predictions for {dataset_name}: {e}")

def log_model_evaluation_metrics(metrics: Dict[str, float], dataset_name: str) -> None:
    """
    Log model evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        dataset_name: Name of the dataset
    """
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(f"{dataset_name}_{metric_name}", metric_value)

def create_experiment_run_name(pipeline_step: str, model_name: str = None) -> str:
    """
    Create a standardized run name for MLflow experiments.
    
    Args:
        pipeline_step: Name of the pipeline step
        model_name: Name of the model (optional)
    
    Returns:
        Formatted run name
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_name:
        return f"{pipeline_step}_{model_name}_{timestamp}"
    else:
        return f"{pipeline_step}_{timestamp}"

def log_dataframe_as_artifact(df: pd.DataFrame, artifact_name: str) -> None:
    """
    Log a DataFrame as an artifact to MLflow.
    
    Args:
        df: DataFrame to log
        artifact_name: Name of the artifact
    """
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            df.to_csv(tmp_file.name, index=True)
            tmp_file_path = tmp_file.name
        
        mlflow.log_artifact(tmp_file_path, artifact_name)
        
        try:
            os.unlink(tmp_file_path)
        except PermissionError:
            pass
        
    except Exception as e:
        log.error(f"Failed to log DataFrame as artifact {artifact_name}: {e}") 