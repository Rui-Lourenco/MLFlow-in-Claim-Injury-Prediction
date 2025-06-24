import logging
import pandas as pd
import numpy as np
import tempfile
import os
import sys
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.mlflow_in_claim_injury_prediction.utils.mlflow_utils import (
    log_dataset_info, create_experiment_run_name
)

log = logging.getLogger(__name__)

def select_features_xgboost(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    threshold: str = "median",
    max_features: int = None
) -> tuple:
    """
    Select features using XGBoost feature importance.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        y_train: Training labels
        y_val: Validation labels
        threshold: Threshold for feature selection ('median', 'mean', or float)
        max_features: Maximum number of features to select
    
    Returns:
        Tuple of (X_train_selected, X_val_selected, X_test_selected, selected_features, feature_importance)
    """
    # Create descriptive run name
    run_name = create_experiment_run_name("feature_selection_xgboost")
    
    log.info(f"Starting XGBoost feature selection run")
    
    log.info("Performing feature selection using XGBoost...")
    
    # Log parameters
    mlflow.log_param("threshold", threshold)
    mlflow.log_param("max_features", max_features)
    
    # Train XGBoost model for feature importance
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    xgb_model.fit(X_train, y_train)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Log feature importance
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        feature_importance.to_csv(tmp_file.name, index=False)
        tmp_file_path = tmp_file.name
    
    # Log the artifact after the file is closed
    mlflow.log_artifact(tmp_file_path, "xgb_feature_importance.csv")
    
    # Clean up the temporary file
    try:
        os.unlink(tmp_file_path)
    except PermissionError:
        # File might still be in use, ignore the error
        pass
    
    # Select features
    if max_features:
        selected_features = feature_importance.head(max_features)['feature'].tolist()
    else:
        selector = SelectFromModel(xgb_model, threshold=threshold)
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.get_support()].tolist()
    
    # Filter datasets
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]
    
    # Log metrics
    mlflow.log_metric("features_selected", len(selected_features))
    mlflow.log_metric("features_original", len(X_train.columns))
    mlflow.log_metric("reduction_percentage", (1 - len(selected_features)/len(X_train.columns)) * 100)
    
    log.info(f"XGBoost selected {len(selected_features)} features out of {len(X_train.columns)}")
    
    return X_train_selected, X_val_selected, X_test_selected, selected_features, feature_importance

def select_features_random_forest(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    threshold: str = "median",
    max_features: int = None
) -> tuple:
    """
    Select features using Random Forest feature importance.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        y_train: Training labels
        y_val: Validation labels
        threshold: Threshold for feature selection ('median', 'mean', or float)
        max_features: Maximum number of features to select
    
    Returns:
        Tuple of (X_train_selected, X_val_selected, X_test_selected, selected_features, feature_importance)
    """
    # Create descriptive run name
    run_name = create_experiment_run_name("feature_selection_rf")
    
    log.info(f"Starting Random Forest feature selection run")
    
    log.info("Performing feature selection using Random Forest...")
    
    # Log parameters
    mlflow.log_param("threshold", threshold)
    mlflow.log_param("max_features", max_features)
    
    # Train Random Forest model for feature importance
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Log feature importance
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        feature_importance.to_csv(tmp_file.name, index=False)
        tmp_file_path = tmp_file.name
    
    # Log the artifact after the file is closed
    mlflow.log_artifact(tmp_file_path, "rf_feature_importance.csv")
    
    # Clean up the temporary file
    try:
        os.unlink(tmp_file_path)
    except PermissionError:
        # File might still be in use, ignore the error
        pass
    
    # Select features
    if max_features:
        selected_features = feature_importance.head(max_features)['feature'].tolist()
    else:
        selector = SelectFromModel(rf_model, threshold=threshold)
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.get_support()].tolist()
    
    # Filter datasets
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]
    
    # Log metrics
    mlflow.log_metric("features_selected", len(selected_features))
    mlflow.log_metric("features_original", len(X_train.columns))
    mlflow.log_metric("reduction_percentage", (1 - len(selected_features)/len(X_train.columns)) * 100)
    
    log.info(f"Random Forest selected {len(selected_features)} features out of {len(X_train.columns)}")
    
    return X_train_selected, X_val_selected, X_test_selected, selected_features, feature_importance

def combine_feature_selection(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    xgb_threshold: str = "median",
    rf_threshold: str = "median",
    max_features: int = None,
    selection_method: str = "union"
) -> tuple:
    """
    Combine feature selection from both XGBoost and Random Forest.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        y_train: Training labels
        y_val: Validation labels
        xgb_threshold: XGBoost threshold for feature selection
        rf_threshold: Random Forest threshold for feature selection
        max_features: Maximum number of features to select
        selection_method: How to combine features ('union', 'intersection', 'xgboost_only', 'rf_only')
    
    Returns:
        Tuple of (X_train_selected, X_val_selected, X_test_selected, selected_features, feature_importance_summary)
    """
    # Create descriptive run name
    run_name = create_experiment_run_name("feature_selection_combined")
    
    log.info(f"Starting combined feature selection run")
    
    log.info("Combining feature selection from XGBoost and Random Forest...")
    
    # Log parameters
    mlflow.log_param("selection_method", selection_method)
    mlflow.log_param("xgb_threshold", xgb_threshold)
    mlflow.log_param("rf_threshold", rf_threshold)
    mlflow.log_param("max_features", max_features)
    
    # Get XGBoost selection
    X_train_xgb, X_val_xgb, X_test_xgb, features_xgb, importance_xgb = select_features_xgboost(
        X_train, X_val, X_test, y_train, y_val, xgb_threshold, max_features
    )
    
    # Get Random Forest selection
    X_train_rf, X_val_rf, X_test_rf, features_rf, importance_rf = select_features_random_forest(
        X_train, X_val, X_test, y_train, y_val, rf_threshold, max_features
    )
    
    # Combine features based on method
    if selection_method == "union":
        selected_features = list(set(features_xgb + features_rf))
    elif selection_method == "intersection":
        selected_features = list(set(features_xgb) & set(features_rf))
    elif selection_method == "xgboost_only":
        selected_features = features_xgb
    elif selection_method == "rf_only":
        selected_features = features_rf
    else:
        selected_features = list(set(features_xgb + features_rf))  # Default to union
    
    # Create feature importance summary
    importance_summary = pd.DataFrame({
        'feature': X_train.columns,
        'xgboost_importance': importance_xgb.set_index('feature')['importance'],
        'rf_importance': importance_rf.set_index('feature')['importance']
    }).fillna(0)
    
    importance_summary['avg_importance'] = (importance_summary['xgboost_importance'] + 
                                           importance_summary['rf_importance']) / 2
    importance_summary = importance_summary.sort_values('avg_importance', ascending=False)
    
    # Filter datasets
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]
    
    # Log metrics
    mlflow.log_metric("features_selected", len(selected_features))
    mlflow.log_metric("features_original", len(X_train.columns))
    mlflow.log_metric("reduction_percentage", (1 - len(selected_features)/len(X_train.columns)) * 100)
    
    # Log feature importance summary
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        importance_summary.to_csv(tmp_file.name, index=False)
        tmp_file_path = tmp_file.name
    
    # Log the artifact after the file is closed
    mlflow.log_artifact(tmp_file_path, "feature_importance_summary.csv")
    
    # Clean up the temporary file
    try:
        os.unlink(tmp_file_path)
    except PermissionError:
        # File might still be in use, ignore the error
        pass
    
    log.info(f"Combined feature selection completed. Selected {len(selected_features)} features using {selection_method} method.")
    
    return X_train_selected, X_val_selected, X_test_selected, selected_features, importance_summary 