import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
import logging
import joblib
import os
import sys

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.mlflow_in_claim_injury_prediction.utils.mlflow_utils import (
    log_dataset_info, log_model_with_metadata,
    log_feature_importance, log_predictions_with_metadata, log_model_evaluation_metrics,
    create_experiment_run_name, log_dataframe_as_artifact
)

log = logging.getLogger(__name__)

def train_xgboost_model(X_train, y_train, X_val, y_val, params=None):
    """
    Train XGBoost model with MLFlow tracking
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        params: XGBoost parameters
    
    Returns:
        Trained XGBoost model and feature importance
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'mlogloss'
        }
    
    # Get experiment ID for nested runs
    experiment = mlflow.get_experiment_by_name("claim_injury_prediction")
    experiment_id = experiment.experiment_id if experiment else None
    
    with mlflow.start_run(run_name=create_experiment_run_name("model_training", "xgboost"), tags={"model_type": "xgboost"}, nested=True) as run:
        log.info(f"Starting XGBoost training run: {run.info.run_id}")
        
        # Log dataset information
        log_dataset_info(X_train, "train", "Training dataset")
        log_dataset_info(X_val, "validation", "Validation dataset")
        
        # Log target distribution
        train_target_dist = y_train.value_counts()
        val_target_dist = y_val.value_counts()
        log_dataframe_as_artifact(train_target_dist.to_frame(), "train_target_distribution.csv")
        log_dataframe_as_artifact(val_target_dist.to_frame(), "val_target_distribution.csv")
        
        # Train model
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_train_proba = model.predict_proba(X_train)
        y_val_proba = model.predict_proba(X_val)
        
        # Calculate metrics
        train_metrics = {
            'f1_score': f1_score(y_train, y_train_pred, average='macro'),
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, average='macro'),
            'recall': recall_score(y_train, y_train_pred, average='macro')
        }
        
        val_metrics = {
            'f1_score': f1_score(y_val, y_val_pred, average='macro'),
            'accuracy': accuracy_score(y_val, y_val_pred),
            'precision': precision_score(y_val, y_val_pred, average='macro'),
            'recall': recall_score(y_val, y_val_pred, average='macro')
        }
        
        # Log metrics
        log_model_evaluation_metrics(train_metrics, "train")
        log_model_evaluation_metrics(val_metrics, "val")
        
        # Log predictions
        log_predictions_with_metadata(y_train.values, y_train_pred, y_train_proba, "train")
        log_predictions_with_metadata(y_val.values, y_val_pred, y_val_proba, "val")
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        log_feature_importance(feature_importance, "xgboost")
        
        # Log model with metadata
        log_model_with_metadata(
            model=model,
            model_name="xgboost",
            model_type="xgboost",
            feature_names=X_train.columns.tolist(),
            model_params=params
        )
        
        # Log classification reports
        train_report = classification_report(y_train, y_train_pred, output_dict=True)
        val_report = classification_report(y_val, y_val_pred, output_dict=True)
        
        log_dataframe_as_artifact(pd.DataFrame(train_report), "train_classification_report.csv")
        log_dataframe_as_artifact(pd.DataFrame(val_report), "val_classification_report.csv")
        
        log.info(f"XGBoost training completed. Train F1: {train_metrics['f1_score']:.4f}, Val F1: {val_metrics['f1_score']:.4f}")
        
        return model, feature_importance

def train_random_forest_model(X_train, y_train, X_val, y_val, params=None):
    """
    Train Random Forest model with MLFlow tracking
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        params: Random Forest parameters
    
    Returns:
        Trained Random Forest model
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
    
    with mlflow.start_run(run_name=create_experiment_run_name("model_training", "random_forest"), tags={"model_type": "random_forest"}, nested=True) as run:
        log.info(f"Starting Random Forest training run: {run.info.run_id}")
        
        # Log dataset information
        log_dataset_info(X_train, "train", "Training dataset")
        log_dataset_info(X_val, "validation", "Validation dataset")
        
        # Log target distribution
        train_target_dist = y_train.value_counts()
        val_target_dist = y_val.value_counts()
        log_dataframe_as_artifact(train_target_dist.to_frame(), "train_target_distribution.csv")
        log_dataframe_as_artifact(val_target_dist.to_frame(), "val_target_distribution.csv")
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_train_proba = model.predict_proba(X_train)
        y_val_proba = model.predict_proba(X_val)
        
        # Calculate metrics
        train_metrics = {
            'f1_score': f1_score(y_train, y_train_pred, average='macro'),
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, average='macro'),
            'recall': recall_score(y_train, y_train_pred, average='macro')
        }
        
        val_metrics = {
            'f1_score': f1_score(y_val, y_val_pred, average='macro'),
            'accuracy': accuracy_score(y_val, y_val_pred),
            'precision': precision_score(y_val, y_val_pred, average='macro'),
            'recall': recall_score(y_val, y_val_pred, average='macro')
        }
        
        # Log metrics
        log_model_evaluation_metrics(train_metrics, "train")
        log_model_evaluation_metrics(val_metrics, "val")
        
        # Log predictions
        log_predictions_with_metadata(y_train.values, y_train_pred, y_train_proba, "train")
        log_predictions_with_metadata(y_val.values, y_val_pred, y_val_proba, "val")
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        log_feature_importance(feature_importance, "random_forest")
        
        # Log model with metadata
        log_model_with_metadata(
            model=model,
            model_name="random_forest",
            model_type="random_forest",
            feature_names=X_train.columns.tolist(),
            model_params=params
        )
        
        # Log classification reports
        train_report = classification_report(y_train, y_train_pred, output_dict=True)
        val_report = classification_report(y_val, y_val_pred, output_dict=True)
        
        log_dataframe_as_artifact(pd.DataFrame(train_report), "train_classification_report.csv")
        log_dataframe_as_artifact(pd.DataFrame(val_report), "val_classification_report.csv")
        
        log.info(f"Random Forest training completed. Train F1: {train_metrics['f1_score']:.4f}, Val F1: {val_metrics['f1_score']:.4f}")
        
        return model, feature_importance

def train_models(X_train, y_train, X_val, y_val, xgb_params=None, rf_params=None):
    """
    Train both XGBoost and Random Forest models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        xgb_params: XGBoost parameters
        rf_params: Random Forest parameters
    
    Returns:
        Dictionary containing trained models and their feature importance
    """
    log.info("Training XGBoost model...")
    xgb_model, xgb_importance = train_xgboost_model(X_train, y_train, X_val, y_val, xgb_params)
    
    log.info("Training Random Forest model...")
    rf_model, rf_importance = train_random_forest_model(X_train, y_train, X_val, y_val, rf_params)
    
    return {
        'xgboost': {
            'model': xgb_model,
            'feature_importance': xgb_importance
        },
        'random_forest': {
            'model': rf_model,
            'feature_importance': rf_importance
        }
    }

def select_best_model(models_dict, X_val, y_val):
    """
    Select the best model based on validation F1 score.
    
    Args:
        models_dict: Dictionary containing trained models
        X_val: Validation features
        y_val: Validation labels
    
    Returns:
        Tuple of (best_model_name, best_model)
    """
    best_score = -1
    best_model_name = None
    best_model = None
    
    for model_name, model_info in models_dict.items():
        model = model_info['model']
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='macro')
        
        if f1 > best_score:
            best_score = f1
            best_model_name = model_name
            best_model = model
    
    log.info(f"Best model: {best_model_name} with F1 score: {best_score:.4f}")
    return best_model_name, best_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Tuple of (metrics_dict, classification_report_dict)
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    metrics = {
        'f1_score': f1_score(y_test, y_pred, average='macro'),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro')
    }
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return metrics, report

def save_model(model, model_name, model_path):
    """
    Save model to disk.
    
    Args:
        model: Trained model
        model_name: Name of the model
        model_path: Path to save the model
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    log.info(f"Model {model_name} saved to {model_path}")

def load_model(filepath):
    """
    Load model from disk.
    
    Args:
        filepath: Path to the saved model
    
    Returns:
        Loaded model
    """
    return joblib.load(filepath)

def predict_proba(model, X):
    """
    Get prediction probabilities.
    
    Args:
        model: Trained model
        X: Features
    
    Returns:
        Prediction probabilities
    """
    return model.predict_proba(X) 