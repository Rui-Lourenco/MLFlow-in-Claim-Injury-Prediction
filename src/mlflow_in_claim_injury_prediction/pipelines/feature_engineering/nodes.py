import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.mlflow_in_claim_injury_prediction.utils.mlflow_utils import (
    log_dataset_info, create_experiment_run_name
)

log = logging.getLogger(__name__)

def engineer_features(
    data: pd.DataFrame,
    create_polynomial_features: bool = False,
    polynomial_degree: int = 2
) -> pd.DataFrame:
    """
    Engineer advanced features from encoded data.
    This includes interaction features, polynomial features, and statistical features.
    
    Args:
        data: Input dataframe (already encoded)
        create_polynomial_features: Whether to create polynomial features
        polynomial_degree: Degree of polynomial features
    
    Returns:
        Engineered dataframe with advanced features
    """
    log.info("Starting feature engineering run")
    
    # Log parameters
    log.info(f"Input data shape: {data.shape}")
    log.info(f"create_polynomial_features: {create_polynomial_features}")
    log.info(f"polynomial_degree: {polynomial_degree}")
    
    # Log dataset information before engineering
    log_dataset_info(data, "data_before_engineering", "Data before engineering")
    
    # Create copy
    data_engineered = data.copy()
    
    log.info("Starting advanced feature engineering...")
    
    # Get numerical columns for feature engineering
    numerical_cols = data_engineered.select_dtypes(include=[np.number]).columns.tolist()
    log.info(f"Found {len(numerical_cols)} numerical columns for feature engineering")
    
    # 1. Create interaction features
    if len(numerical_cols) >= 2:
        data_engineered = _create_interaction_features(data_engineered, numerical_cols)
    
    # 2. Create polynomial features if requested
    if create_polynomial_features and len(numerical_cols) > 0:
        data_engineered = _create_polynomial_features(data_engineered, numerical_cols, polynomial_degree)
    
    # 3. Create statistical features
    if len(numerical_cols) > 0:
        data_engineered = _create_statistical_features(data_engineered, numerical_cols)
    
    # Log final results
    _log_engineering_results(data_engineered, data.shape[1])
    
    return data_engineered

def _create_interaction_features(data: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
    """Create meaningful interaction features between numerical columns."""
    log.info("Creating interaction features...")
    
    # Limit to top numerical features to avoid too many interactions
    top_features = numerical_cols[:5]
    
    interactions_created = 0
    for i, col1 in enumerate(top_features):
        for col2 in numerical_cols[i+1:6]:  # Limit to 6 features total
            interaction_name = f"{col1}_x_{col2}"
            data[interaction_name] = data[col1] * data[col2]
            interactions_created += 1
    
    log.info(f"Created {interactions_created} interaction features")
    
    return data

def _create_polynomial_features(data: pd.DataFrame, numerical_cols: list, degree: int) -> pd.DataFrame:
    """Create polynomial features for numerical columns."""
    log.info(f"Creating polynomial features with degree {degree}...")
    
    # Limit to top numerical features to avoid explosion
    top_features = numerical_cols[:10]
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    # Fit and transform
    data_poly = poly.fit_transform(data[top_features])
    
    # Create feature names
    poly_feature_names = [f"poly_{i}" for i in range(data_poly.shape[1])]
    
    # Add polynomial features to dataframe
    for i, name in enumerate(poly_feature_names):
        data[name] = data_poly[:, i]
    
    log.info(f"Created {len(poly_feature_names)} polynomial features")
    
    return data

def _create_statistical_features(data: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
    """Create statistical features from numerical columns."""
    log.info("Creating statistical features...")
    
    # Mean of numerical features
    data['numerical_mean'] = data[numerical_cols].mean(axis=1)
    
    # Standard deviation of numerical features
    data['numerical_std'] = data[numerical_cols].std(axis=1)
    
    # Median of numerical features
    data['numerical_median'] = data[numerical_cols].median(axis=1)
    
    # Range of numerical features
    data['numerical_range'] = data[numerical_cols].max(axis=1) - data[numerical_cols].min(axis=1)
    
    log.info("Created statistical features (mean, std, median, range)")
    
    return data

def _log_engineering_results(data: pd.DataFrame, original_features: int):
    """Log final engineering results."""
    final_features = data.shape[1]
    new_features = final_features - original_features
    
    log.info(f"Feature engineering completed successfully")
    log.info(f"Original features: {original_features}")
    log.info(f"Final features: {final_features}")
    log.info(f"New features created: {new_features}")
    
    log_dataset_info(data, "data_after_engineering", "Data after engineering") 