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
        Trained XGBoost model
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
    
    with mlflow.start_run(run_name="xgboost_training"):
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        train_f1 = f1_score(y_train, y_train_pred, average='macro')
        val_f1 = f1_score(y_val, y_val_pred, average='macro')
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        # Log metrics
        mlflow.log_metric("train_f1_score", train_f1)
        mlflow.log_metric("val_f1_score", val_f1)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("val_accuracy", val_acc)
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        mlflow.log_artifact(feature_importance.to_csv(index=False), "feature_importance_xgboost.csv")
        
        # Log model
        mlflow.xgboost.log_model(model, "model")
        
        log.info(f"XGBoost - Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
        
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
    
    with mlflow.start_run(run_name="random_forest_training"):
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        train_f1 = f1_score(y_train, y_train_pred, average='macro')
        val_f1 = f1_score(y_val, y_val_pred, average='macro')
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        # Log metrics
        mlflow.log_metric("train_f1_score", train_f1)
        mlflow.log_metric("val_f1_score", val_f1)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("val_accuracy", val_acc)
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        mlflow.log_artifact(feature_importance.to_csv(index=False), "feature_importance_rf.csv")
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        log.info(f"Random Forest - Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
        
        return model, feature_importance

def train_models(X_train, y_train, X_val, y_val, xgb_params=None, rf_params=None):
    """
    Train both XGBoost and Random Forest models
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        xgb_params: XGBoost parameters
        rf_params: Random Forest parameters
    
    Returns:
        Dictionary containing trained models and feature importance
    """
    results = {}
    
    # Train XGBoost
    xgb_model, xgb_importance = train_xgboost_model(X_train, y_train, X_val, y_val, xgb_params)
    results['xgboost'] = {
        'model': xgb_model,
        'feature_importance': xgb_importance
    }
    
    # Train Random Forest
    rf_model, rf_importance = train_random_forest_model(X_train, y_train, X_val, y_val, rf_params)
    results['random_forest'] = {
        'model': rf_model,
        'feature_importance': rf_importance
    }
    
    return results

def select_best_model(models_dict, X_val, y_val):
    """
    Select the best model based on validation F1 score
    
    Args:
        models_dict: Dictionary containing trained models
        X_val: Validation features
        y_val: Validation labels
    
    Returns:
        Best model name and model object
    """
    best_score = 0
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

def save_model(model, model_name, filepath):
    """
    Save model to disk
    
    Args:
        model: Trained model
        model_name: Name of the model
        filepath: Path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    log.info(f"Model {model_name} saved to {filepath}")

def load_model(filepath):
    """
    Load model from disk
    
    Args:
        filepath: Path to the saved model
    
    Returns:
        Loaded model
    """
    model = joblib.load(filepath)
    log.info(f"Model loaded from {filepath}")
    return model

def predict(model, X):
    """
    Make predictions using the trained model
    
    Args:
        model: Trained model
        X: Features for prediction
    
    Returns:
        Predictions
    """
    return model.predict(X)

def predict_proba(model, X):
    """
    Make probability predictions using the trained model
    
    Args:
        model: Trained model
        X: Features for prediction
    
    Returns:
        Probability predictions
    """
    return model.predict_proba(X)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='macro'),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro')
    }
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    log.info(f"Model Evaluation - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
    
    return metrics, report 