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

def setup_mlflow(experiment_name: str = "claim_injury_prediction") -> None:
    """
    Setup MLflow tracking and experiment.
    
    Args:
        experiment_name: Name of the MLflow experiment
    """
    try:
        # Set tracking URI to local filesystem
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        log.info(f"MLflow setup complete. Experiment: {experiment_name}")
        
    except Exception as e:
        log.error(f"Failed to setup MLflow: {e}")
        raise

def start_mlflow_run(run_name: str, tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
    """
    Start an MLflow run with proper naming and tags.
    
    Args:
        run_name: Name for the MLflow run
        tags: Additional tags for the run
        
    Returns:
        Active MLflow run
    """
    if tags is None:
        tags = {}
    
    # Add default tags
    default_tags = {
        "project": "claim_injury_prediction",
        "version": "1.0",
        "timestamp": datetime.now().isoformat()
    }
    tags.update(default_tags)
    
    return mlflow.start_run(run_name=run_name, tags=tags)

def log_dataset_info(data: pd.DataFrame, dataset_name: str, description: str = "") -> None:
    """
    Log dataset information to MLflow.
    
    Args:
        data: DataFrame to log
        dataset_name: Name of the dataset
        description: Description of the dataset
    """
    try:
        # Log dataset shape
        mlflow.log_param(f"{dataset_name}_shape", str(data.shape))
        mlflow.log_param(f"{dataset_name}_columns", len(data.columns))
        mlflow.log_param(f"{dataset_name}_rows", len(data))
        
        # Log dataset statistics
        if len(data.select_dtypes(include=[np.number]).columns) > 0:
            numeric_stats = data.describe()
            # Save to temporary file first
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                numeric_stats.to_csv(tmp_file.name)
                mlflow.log_artifact(tmp_file.name, f"{dataset_name}_statistics.csv")
                os.unlink(tmp_file.name)  # Clean up
        
        # Log missing values
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            missing_df = pd.DataFrame({
                'column': missing_values.index,
                'missing_count': missing_values.values,
                'missing_percentage': (missing_values.values / len(data)) * 100
            })
            # Save to temporary file first
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                missing_df.to_csv(tmp_file.name, index=False)
                mlflow.log_artifact(tmp_file.name, f"{dataset_name}_missing_values.csv")
                os.unlink(tmp_file.name)  # Clean up
        
        # Log dataset sample
        sample_data = data.head(100)
        # Save to temporary file first
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            sample_data.to_csv(tmp_file.name, index=False)
            mlflow.log_artifact(tmp_file.name, f"{dataset_name}_sample.csv")
            os.unlink(tmp_file.name)  # Clean up
        
        log.info(f"Logged dataset info for {dataset_name}")
        
    except Exception as e:
        log.warning(f"Failed to log dataset info for {dataset_name}: {e}")

def log_model_with_metadata(model, model_name: str, model_type: str, 
                          feature_names: list, model_params: dict) -> None:
    """
    Log model with comprehensive metadata.
    
    Args:
        model: Trained model object
        model_name: Name of the model
        model_type: Type of model (e.g., 'xgboost', 'random_forest')
        feature_names: List of feature names
        model_params: Model parameters
    """
    try:
        # Log model parameters
        for param_name, param_value in model_params.items():
            mlflow.log_param(f"{model_name}_{param_name}", param_value)
        
        # Log model metadata
        mlflow.log_param(f"{model_name}_type", model_type)
        mlflow.log_param(f"{model_name}_features_count", len(feature_names))
        mlflow.log_param(f"{model_name}_feature_names", str(feature_names))
        
        # Log model based on type
        if model_type == "xgboost":
            mlflow.xgboost.log_model(model, f"{model_name}_model")
        elif model_type == "random_forest":
            mlflow.sklearn.log_model(model, f"{model_name}_model")
        else:
            mlflow.sklearn.log_model(model, f"{model_name}_model")
        
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
        # Ensure proper column names
        if 'feature' not in feature_importance.columns or 'importance' not in feature_importance.columns:
            log.warning(f"Feature importance DataFrame for {model_name} doesn't have expected columns")
            return
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # Log top features
        top_features = feature_importance.head(20)
        # Save to temporary file first
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            top_features.to_csv(tmp_file.name, index=False)
            mlflow.log_artifact(tmp_file.name, f"{model_name}_top_features.csv")
            os.unlink(tmp_file.name)  # Clean up
        
        # Log all features
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            feature_importance.to_csv(tmp_file.name, index=False)
            mlflow.log_artifact(tmp_file.name, f"{model_name}_all_features.csv")
            os.unlink(tmp_file.name)  # Clean up
        
        # Log feature importance statistics
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
        log.warning(f"Failed to log feature importance for {model_name}: {e}")

def log_predictions_with_metadata(y_true: np.ndarray, y_pred: np.ndarray, 
                                y_proba: np.ndarray, dataset_name: str) -> None:
    """
    Log predictions with comprehensive metadata.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        dataset_name: Name of the dataset (e.g., 'train', 'val', 'test')
    """
    try:
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'true_label': y_true,
            'predicted_label': y_pred,
            'prediction_confidence': np.max(y_proba, axis=1)
        })
        
        # Add probability columns
        for i in range(y_proba.shape[1]):
            predictions_df[f'prob_class_{i}'] = y_proba[:, i]
        
        # Log predictions
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            predictions_df.to_csv(tmp_file.name, index=False)
            mlflow.log_artifact(tmp_file.name, f"{dataset_name}_predictions.csv")
            os.unlink(tmp_file.name)  # Clean up
        
        # Log prediction statistics
        mlflow.log_metric(f"{dataset_name}_predictions_count", len(y_pred))
        mlflow.log_metric(f"{dataset_name}_unique_predictions", len(np.unique(y_pred)))
        mlflow.log_metric(f"{dataset_name}_avg_confidence", predictions_df['prediction_confidence'].mean())
        
        log.info(f"Logged predictions for {dataset_name}")
        
    except Exception as e:
        log.warning(f"Failed to log predictions for {dataset_name}: {e}")

def log_model_evaluation_metrics(metrics: Dict[str, float], dataset_name: str) -> None:
    """
    Log model evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        dataset_name: Name of the dataset
    """
    try:
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(f"{dataset_name}_{metric_name}", metric_value)
        
        log.info(f"Logged {len(metrics)} metrics for {dataset_name}")
        
    except Exception as e:
        log.warning(f"Failed to log metrics for {dataset_name}: {e}")

def create_experiment_run_name(pipeline_step: str, model_name: str = None) -> str:
    """
    Create a descriptive run name for MLflow.
    
    Args:
        pipeline_step: Name of the pipeline step
        model_name: Name of the model (optional)
        
    Returns:
        Descriptive run name
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if model_name:
        return f"{pipeline_step}_{model_name}_{timestamp}"
    else:
        return f"{pipeline_step}_{timestamp}"

def log_dataframe_as_artifact(df: pd.DataFrame, artifact_name: str) -> None:
    """
    Log a DataFrame as an artifact by saving it to a temporary file first.
    
    Args:
        df: DataFrame to log
        artifact_name: Name for the artifact
    """
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            mlflow.log_artifact(tmp_file.name, artifact_name)
            os.unlink(tmp_file.name)  # Clean up
        log.info(f"Logged DataFrame as artifact: {artifact_name}")
    except Exception as e:
        log.warning(f"Failed to log DataFrame as artifact {artifact_name}: {e}") 