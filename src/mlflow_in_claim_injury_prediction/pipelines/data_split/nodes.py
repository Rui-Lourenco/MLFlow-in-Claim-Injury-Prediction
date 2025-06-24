import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.mlflow_in_claim_injury_prediction.utils.mlflow_utils import (
    setup_mlflow, start_mlflow_run, log_dataset_info, create_experiment_run_name, log_dataframe_as_artifact
)

log = logging.getLogger(__name__)

def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.1,
    val_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Split data into train, validation, and test sets.
    
    First splits into temp_data (90%) and test (10%), then splits temp_data into train (80%) and validation (20%).
    Assumes target column has been cleaned of NaN values in previous steps.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        test_size: Proportion of data for test set (default: 0.1)
        val_size: Proportion of temp_data for validation set (default: 0.2)
        random_state: Random state for reproducibility
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Setup MLflow
    setup_mlflow()
    
    # Create descriptive run name
    run_name = create_experiment_run_name("data_split")
    
    with start_mlflow_run(run_name=run_name, tags={"pipeline": "data_split"}) as run:
        log.info(f"Starting data split run: {run.info.run_id}")
        
        # Log parameters
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("val_size", val_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("target_column", target_column)
        
        # Log original dataset info
        log_dataset_info(df, "original_data", "Original dataset before splitting")
        
        # Log target distribution
        target_dist = df[target_column].value_counts()
        log_dataframe_as_artifact(target_dist.to_frame(), "original_target_distribution.csv")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Check for NaN values in features (these will be handled in transformations)
        feature_nan_counts = X.isnull().sum()
        if feature_nan_counts.sum() > 0:
            log.info(f"Found NaN values in features (will be handled in transformations):")
            for col, count in feature_nan_counts[feature_nan_counts > 0].items():
                log.info(f"  {col}: {count} NaN values")
            mlflow.log_metric("total_feature_nan_values", feature_nan_counts.sum())
        
        # First split: temp_data (90%) and test (10%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train (80% of temp_data) and validation (20% of temp_data)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        
        # Log data split information
        log.info(f"Data split completed:")
        log.info(f"  Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
        log.info(f"  Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(df)*100:.1f}%)")
        log.info(f"  Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
        
        # Log metrics
        mlflow.log_metric("train_samples", X_train.shape[0])
        mlflow.log_metric("val_samples", X_val.shape[0])
        mlflow.log_metric("test_samples", X_test.shape[0])
        mlflow.log_metric("total_features", X_train.shape[1])
        mlflow.log_metric("train_percentage", X_train.shape[0]/len(df)*100)
        mlflow.log_metric("val_percentage", X_val.shape[0]/len(df)*100)
        mlflow.log_metric("test_percentage", X_test.shape[0]/len(df)*100)
        
        # Log split datasets info
        log_dataset_info(X_train, "train_features", "Training features")
        log_dataset_info(X_val, "val_features", "Validation features")
        log_dataset_info(X_test, "test_features", "Test features")
        
        # Log target distributions for each split
        train_target_dist = y_train.value_counts()
        val_target_dist = y_val.value_counts()
        test_target_dist = y_test.value_counts()
        
        log_dataframe_as_artifact(train_target_dist.to_frame(), "train_target_distribution.csv")
        log_dataframe_as_artifact(val_target_dist.to_frame(), "val_target_distribution.csv")
        log_dataframe_as_artifact(test_target_dist.to_frame(), "test_target_distribution.csv")
        
        # Log class balance metrics
        mlflow.log_metric("train_class_balance", train_target_dist.min() / train_target_dist.max())
        mlflow.log_metric("val_class_balance", val_target_dist.min() / val_target_dist.max())
        mlflow.log_metric("test_class_balance", test_target_dist.min() / test_target_dist.max())
        
        return X_train, X_val, X_test, y_train, y_val, y_test 