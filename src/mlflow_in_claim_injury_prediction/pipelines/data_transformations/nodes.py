import logging
import pandas as pd
import numpy as np
import mlflow
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.utils import NA_imputer, create_new_features
from src.mlflow_in_claim_injury_prediction.utils.mlflow_utils import (
    log_dataset_info, create_experiment_run_name
)

log = logging.getLogger(__name__)

def apply_processing(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame
) -> tuple:
    """
    Apply advanced processing techniques from utils.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
    
    Returns:
        Tuple of processed (X_train, X_val, X_test)
    """
    log.info("Starting data processing run")
    
    log.info("Applying advanced processing techniques...")
    
    # Create copies
    X_train_processed = X_train.copy()
    X_val_processed = X_val.copy()
    X_test_processed = X_test.copy()
    
    # Apply NA imputation
    NA_imputer(X_train_processed, X_val_processed)
    NA_imputer(X_train_processed, X_test_processed)
    
    # Create new features
    create_new_features(X_train_processed, X_val_processed)
    create_new_features(X_train_processed, X_test_processed)
    
    # Log processing results
    mlflow.log_metric("train_features_after_processing", X_train_processed.shape[1])
    mlflow.log_metric("val_features_after_processing", X_val_processed.shape[1])
    mlflow.log_metric("test_features_after_processing", X_test_processed.shape[1])
    
    log.info("Data processing completed")
    
    return X_train_processed, X_val_processed, X_test_processed 