import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow

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
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        test_size: Proportion of data for test set (default: 0.1)
        val_size: Proportion of temp_data for validation set (default: 0.2)
        random_state: Random state for reproducibility
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Log parameters
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("val_size", val_size)
    mlflow.log_param("random_state", random_state)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
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
    
    return X_train, X_val, X_test, y_train, y_val, y_test 