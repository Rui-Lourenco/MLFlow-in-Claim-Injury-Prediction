import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
import mlflow

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
    log.info("Performing feature selection using XGBoost...")
    
    # Train XGBoost model for feature importance
    xgb_model = XGBClassifier(
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
    mlflow.log_artifact(feature_importance.to_csv(index=False), "xgboost_feature_importance.csv")
    
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
    log.info("Performing feature selection using Random Forest...")
    
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
    mlflow.log_artifact(feature_importance.to_csv(index=False), "rf_feature_importance.csv")
    
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
    log.info("Combining feature selection from XGBoost and Random Forest...")
    
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
    mlflow.log_param("selection_method", selection_method)
    mlflow.log_param("xgb_threshold", xgb_threshold)
    mlflow.log_param("rf_threshold", rf_threshold)
    mlflow.log_metric("features_selected", len(selected_features))
    mlflow.log_metric("features_original", len(X_train.columns))
    mlflow.log_metric("reduction_percentage", (1 - len(selected_features)/len(X_train.columns)) * 100)
    
    # Log feature importance summary
    mlflow.log_artifact(importance_summary.to_csv(index=False), "feature_importance_summary.csv")
    
    log.info(f"Combined feature selection completed. Selected {len(selected_features)} features using {selection_method} method.")
    
    return X_train_selected, X_val_selected, X_test_selected, selected_features, importance_summary 