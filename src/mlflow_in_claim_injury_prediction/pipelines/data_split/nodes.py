import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.mlflow_in_claim_injury_prediction.utils.mlflow_utils import (
    log_dataset_info, log_dataframe_as_artifact
)

log = logging.getLogger(__name__)

def _find_target_column(data: pd.DataFrame, target_column: str) -> str:
    """Find the target column with multiple name variations."""
    possible_names = [
        target_column,
        target_column.lower(),
        target_column.upper(),
        target_column.replace('_', ' '),
        target_column.replace(' ', '_'),
        'claim_injury_type',
        'Claim Injury Type',
        'CLAIM_INJURY_TYPE'
    ]
    
    for col_name in possible_names:
        if col_name in data.columns:
            log.info(f"Found target column for splitting: '{col_name}'")
            return col_name
    
    log.error(f"Target column not found. Tried: {possible_names}")
    log.error(f"Available columns: {list(data.columns)}")
    raise ValueError(f"Target column '{target_column}' not found in data")

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
    log.info("Starting data split run")
    
    # Find the actual target column name
    actual_target_column = _find_target_column(df, target_column)
    
    # Log parameters
    log.info(f"test_size: {test_size}")
    log.info(f"val_size: {val_size}")
    log.info(f"random_state: {random_state}")
    log.info(f"target_column: {actual_target_column}")
    
    # Log original dataset info
    log_dataset_info(df, "original_data", "Original dataset before splitting")
    
    # Log target distribution
    target_dist = df[actual_target_column].value_counts()
    log_dataframe_as_artifact(target_dist.to_frame(), "original_target_distribution.csv")
    
    # Separate features and target
    X = df.drop(columns=[actual_target_column])
    y = df[actual_target_column]
    
    # Check for NaN values in features (these will be handled in transformations)
    feature_nan_counts = X.isnull().sum()
    if feature_nan_counts.sum() > 0:
        log.info(f"Found NaN values in features (will be handled in transformations):")
        for col, count in feature_nan_counts[feature_nan_counts > 0].items():
            log.info(f"  {col}: {count} NaN values")
        log.info(f"total_feature_nan_values: {feature_nan_counts.sum()}")
    
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
    log.info(f"train_samples: {X_train.shape[0]}")
    log.info(f"val_samples: {X_val.shape[0]}")
    log.info(f"test_samples: {X_test.shape[0]}")
    log.info(f"total_features: {X_train.shape[1]}")
    log.info(f"train_percentage: {X_train.shape[0]/len(df)*100:.1f}%")
    log.info(f"val_percentage: {X_val.shape[0]/len(df)*100:.1f}%")
    log.info(f"test_percentage: {X_test.shape[0]/len(df)*100:.1f}%")
    
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
    log.info(f"train_class_balance: {train_target_dist.min() / train_target_dist.max():.2f}")
    log.info(f"val_class_balance: {val_target_dist.min() / val_target_dist.max():.2f}")
    log.info(f"test_class_balance: {test_target_dist.min() / test_target_dist.max():.2f}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test 