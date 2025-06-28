import logging
import pandas as pd
import tempfile
import os
import sys
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb

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
    log.info(f"Starting XGBoost feature selection run")
    
    log.info("Performing feature selection using XGBoost...")
    
    mlflow.log_param("threshold", threshold)
    mlflow.log_param("max_features", max_features)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    xgb_model.fit(X_train, y_train)
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        feature_importance.to_csv(tmp_file.name, index=False)
        tmp_file_path = tmp_file.name
    
    mlflow.log_artifact(tmp_file_path, "xgb_feature_importance.csv")
    
    try:
        os.unlink(tmp_file_path)
    except PermissionError:
        pass
    
    if max_features:
        selected_features = feature_importance.head(max_features)['feature'].tolist()
    else:
        selector = SelectFromModel(xgb_model, threshold=threshold)
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.get_support()].tolist()
    
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]
    
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
    log.info(f"Starting Random Forest feature selection run")
    
    log.info("Performing feature selection using Random Forest...")
    
    mlflow.log_param("threshold", threshold)
    mlflow.log_param("max_features", max_features)
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        feature_importance.to_csv(tmp_file.name, index=False)
        tmp_file_path = tmp_file.name
    
    mlflow.log_artifact(tmp_file_path, "rf_feature_importance.csv")
    
    try:
        os.unlink(tmp_file_path)
    except PermissionError:
        pass
    
    if max_features:
        selected_features = feature_importance.head(max_features)['feature'].tolist()
    else:
        selector = SelectFromModel(rf_model, threshold=threshold)
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.get_support()].tolist()
    
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]
    
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
        xgb_threshold: XGBoost threshold
        rf_threshold: Random Forest threshold
        max_features: Maximum number of features
        selection_method: 'union' or 'intersection'
    
    Returns:
        Tuple of selected datasets and feature information
    """
    log.info("Starting combined feature selection...")
    
    mlflow.log_param("selection_method", selection_method)
    mlflow.log_param("xgb_threshold", xgb_threshold)
    mlflow.log_param("rf_threshold", rf_threshold)
    
    xgb_train, xgb_val, xgb_test, xgb_features, xgb_importance = select_features_xgboost(
        X_train, X_val, X_test, y_train, y_val, xgb_threshold, max_features
    )
    
    rf_train, rf_val, rf_test, rf_features, rf_importance = select_features_random_forest(
        X_train, X_val, X_test, y_train, y_val, rf_threshold, max_features
    )
    
    if selection_method == "union":
        selected_features = list(set(xgb_features) | set(rf_features))
    elif selection_method == "intersection":
        selected_features = list(set(xgb_features) & set(rf_features))
    else:
        raise ValueError("selection_method must be 'union' or 'intersection'")
    
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]
    
    feature_importance_summary = pd.DataFrame({
        'feature': selected_features,
        'xgb_importance': [xgb_importance[xgb_importance['feature'] == f]['importance'].iloc[0] 
                          if f in xgb_features else 0 for f in selected_features],
        'rf_importance': [rf_importance[rf_importance['feature'] == f]['importance'].iloc[0] 
                         if f in rf_features else 0 for f in selected_features]
    })
    
    feature_importance_summary['avg_importance'] = (
        feature_importance_summary['xgb_importance'] + feature_importance_summary['rf_importance']
    ) / 2
    
    feature_importance_summary = feature_importance_summary.sort_values('avg_importance', ascending=False)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        feature_importance_summary.to_csv(tmp_file.name, index=False)
        tmp_file_path = tmp_file.name
    
    mlflow.log_artifact(tmp_file_path, "combined_feature_importance.csv")
    
    try:
        os.unlink(tmp_file_path)
    except PermissionError:
        pass
    
    mlflow.log_metric("final_features_selected", len(selected_features))
    mlflow.log_metric("xgb_features", len(xgb_features))
    mlflow.log_metric("rf_features", len(rf_features))
    
    log.info(f"Combined selection: {len(selected_features)} features selected")
    log.info(f"XGBoost features: {len(xgb_features)}, RF features: {len(rf_features)}")
    
    return X_train_selected, X_val_selected, X_test_selected, selected_features, feature_importance_summary 