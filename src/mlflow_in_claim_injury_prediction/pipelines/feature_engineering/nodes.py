import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import mlflow
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.utils import extract_dates_components, flag_public_holiday_accidents, flag_weekend_accidents, get_season

log = logging.getLogger(__name__)

def engineer_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    date_columns: list = None,
    create_polynomial_features: bool = False,
    polynomial_degree: int = 2
) -> tuple:
    """
    Engineer new features from existing data.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        date_columns: List of date column names
        create_polynomial_features: Whether to create polynomial features
        polynomial_degree: Degree of polynomial features
    
    Returns:
        Tuple of (X_train_engineered, X_val_engineered, X_test_engineered, feature_names)
    """
    # Log parameters
    mlflow.log_param("create_polynomial_features", create_polynomial_features)
    mlflow.log_param("polynomial_degree", polynomial_degree)
    mlflow.log_param("date_columns_count", len(date_columns) if date_columns else 0)
    
    # Create copies
    X_train_engineered = X_train.copy()
    X_val_engineered = X_val.copy()
    X_test_engineered = X_test.copy()
    
    log.info("Starting feature engineering...")
    
    # Extract date components if date columns are provided
    if date_columns:
        log.info("Extracting date components...")
        
        # Find actual date columns in the data
        available_date_columns = [col for col in date_columns if col in X_train_engineered.columns]
        
        if available_date_columns:
            # Extract date components
            X_train_engineered = extract_dates_components([X_train_engineered], available_date_columns)[0]
            X_val_engineered = extract_dates_components([X_val_engineered], available_date_columns)[0]
            X_test_engineered = extract_dates_components([X_test_engineered], available_date_columns)[0]
            
            log.info(f"Extracted date components from {len(available_date_columns)} date columns")
    
    # Create interaction features for numerical columns
    numerical_cols = X_train_engineered.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) >= 2:
        log.info("Creating interaction features...")
        
        # Create some meaningful interactions
        for i, col1 in enumerate(numerical_cols[:5]):  # Limit to first 5 to avoid too many features
            for col2 in numerical_cols[i+1:6]:
                interaction_name = f"{col1}_x_{col2}"
                X_train_engineered[interaction_name] = X_train_engineered[col1] * X_train_engineered[col2]
                X_val_engineered[interaction_name] = X_val_engineered[col1] * X_val_engineered[col2]
                X_test_engineered[interaction_name] = X_test_engineered[col1] * X_test_engineered[col2]
        
        log.info("Created interaction features")
    
    # Create polynomial features if requested
    if create_polynomial_features and len(numerical_cols) > 0:
        log.info(f"Creating polynomial features with degree {polynomial_degree}...")
        
        # Limit to top numerical features to avoid explosion
        top_numerical = numerical_cols[:10]  # Limit to 10 features
        
        poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
        
        # Fit on training data
        X_train_poly = poly.fit_transform(X_train_engineered[top_numerical])
        X_val_poly = poly.transform(X_val_engineered[top_numerical])
        X_test_poly = poly.transform(X_test_engineered[top_numerical])
        
        # Create feature names
        poly_feature_names = [f"poly_{i}" for i in range(X_train_poly.shape[1])]
        
        # Add polynomial features to dataframes
        for i, name in enumerate(poly_feature_names):
            X_train_engineered[name] = X_train_poly[:, i]
            X_val_engineered[name] = X_val_poly[:, i]
            X_test_engineered[name] = X_test_poly[:, i]
        
        log.info(f"Created {len(poly_feature_names)} polynomial features")
    
    # Create statistical features
    if len(numerical_cols) > 0:
        log.info("Creating statistical features...")
        
        # Mean of numerical features
        X_train_engineered['numerical_mean'] = X_train_engineered[numerical_cols].mean(axis=1)
        X_val_engineered['numerical_mean'] = X_val_engineered[numerical_cols].mean(axis=1)
        X_test_engineered['numerical_mean'] = X_test_engineered[numerical_cols].mean(axis=1)
        
        # Standard deviation of numerical features
        X_train_engineered['numerical_std'] = X_train_engineered[numerical_cols].std(axis=1)
        X_val_engineered['numerical_std'] = X_val_engineered[numerical_cols].std(axis=1)
        X_test_engineered['numerical_std'] = X_test_engineered[numerical_cols].std(axis=1)
        
        log.info("Created statistical features")
    
    # Log feature engineering statistics
    original_features = len(X_train.columns)
    final_features = len(X_train_engineered.columns)
    new_features = final_features - original_features
    
    mlflow.log_metric("original_features", original_features)
    mlflow.log_metric("final_features", final_features)
    mlflow.log_metric("new_features_created", new_features)
    
    log.info(f"Feature engineering completed. Created {new_features} new features.")
    log.info(f"Total features: {final_features}")
    
    return X_train_engineered, X_val_engineered, X_test_engineered, X_train_engineered.columns.tolist() 