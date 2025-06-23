import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
import mlflow
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.utils import NA_imputer, apply_frequency_encoding, create_new_features

log = logging.getLogger(__name__)

def process_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    numerical_features: list,
    categorical_features: list,
    scaling_method: str = "standard",
    imputation_method: str = "median"
) -> tuple:
    """
    Process data by handling missing values and scaling numerical features.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        scaling_method: Scaling method ('standard', 'minmax', or 'none')
        imputation_method: Imputation method ('median', 'mean', 'knn', or 'most_frequent')
    
    Returns:
        Tuple of processed (X_train, X_val, X_test, scaler, imputer)
    """
    # Log parameters
    mlflow.log_param("scaling_method", scaling_method)
    mlflow.log_param("imputation_method", imputation_method)
    mlflow.log_param("numerical_features_count", len(numerical_features))
    mlflow.log_param("categorical_features_count", len(categorical_features))
    
    # Create copies to avoid modifying original data
    X_train_processed = X_train.copy()
    X_val_processed = X_val.copy()
    X_test_processed = X_test.copy()
    
    # Handle missing values
    log.info("Handling missing values...")
    
    # For numerical features
    if numerical_features:
        if imputation_method == "median":
            imputer = SimpleImputer(strategy='median')
        elif imputation_method == "mean":
            imputer = SimpleImputer(strategy='mean')
        elif imputation_method == "knn":
            imputer = KNNImputer(n_neighbors=5)
        else:
            imputer = SimpleImputer(strategy='median')
        
        # Fit on training data
        X_train_processed[numerical_features] = imputer.fit_transform(X_train_processed[numerical_features])
        X_val_processed[numerical_features] = imputer.transform(X_val_processed[numerical_features])
        X_test_processed[numerical_features] = imputer.transform(X_test_processed[numerical_features])
        
        log.info(f"Imputed missing values in {len(numerical_features)} numerical features using {imputation_method}")
    
    # For categorical features
    if categorical_features:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        
        # Fit on training data
        X_train_processed[categorical_features] = cat_imputer.fit_transform(X_train_processed[categorical_features])
        X_val_processed[categorical_features] = cat_imputer.transform(X_val_processed[categorical_features])
        X_test_processed[categorical_features] = cat_imputer.transform(X_test_processed[categorical_features])
        
        log.info(f"Imputed missing values in {len(categorical_features)} categorical features using most_frequent")
    
    # Scale numerical features
    scaler = None
    if numerical_features and scaling_method != "none":
        log.info(f"Scaling numerical features using {scaling_method} scaling...")
        
        if scaling_method == "standard":
            scaler = StandardScaler()
        elif scaling_method == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        
        # Fit on training data
        X_train_processed[numerical_features] = scaler.fit_transform(X_train_processed[numerical_features])
        X_val_processed[numerical_features] = scaler.transform(X_val_processed[numerical_features])
        X_test_processed[numerical_features] = scaler.transform(X_test_processed[numerical_features])
        
        log.info(f"Scaled {len(numerical_features)} numerical features using {scaling_method} scaling")
    
    # Log processing statistics
    mlflow.log_metric("train_missing_values_before", X_train.isnull().sum().sum())
    mlflow.log_metric("val_missing_values_before", X_val.isnull().sum().sum())
    mlflow.log_metric("test_missing_values_before", X_test.isnull().sum().sum())
    
    mlflow.log_metric("train_missing_values_after", X_train_processed.isnull().sum().sum())
    mlflow.log_metric("val_missing_values_after", X_val_processed.isnull().sum().sum())
    mlflow.log_metric("test_missing_values_after", X_test_processed.isnull().sum().sum())
    
    log.info("Data processing completed successfully")
    
    return X_train_processed, X_val_processed, X_test_processed, scaler, imputer

def apply_advanced_processing(
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
    log.info("Applying advanced processing techniques...")
    
    # Create copies
    X_train_processed = X_train.copy()
    X_val_processed = X_val.copy()
    X_test_processed = X_test.copy()
    
    # Apply frequency encoding
    X_train_processed, X_val_processed = apply_frequency_encoding(
        X_train_processed, X_val_processed, save_encoding=False
    )
    X_train_processed, X_test_processed = apply_frequency_encoding(
        X_train_processed, X_test_processed, save_encoding=False
    )
    
    # Apply NA imputation
    NA_imputer(X_train_processed, X_val_processed)
    NA_imputer(X_train_processed, X_test_processed)
    
    # Create new features
    create_new_features(X_train_processed, X_val_processed)
    create_new_features(X_train_processed, X_test_processed)
    
    log.info("Advanced processing completed")
    
    return X_train_processed, X_val_processed, X_test_processed 