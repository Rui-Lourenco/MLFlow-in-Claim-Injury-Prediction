import logging
import pandas as pd
import numpy as np
import mlflow
import sys
import os
import tempfile
import shutil
from typing import Dict, Any, Tuple
from sklearn.metrics import confusion_matrix
from src.mlflow_in_claim_injury_prediction.utils.mlflow_utils import log_dataset_info

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.model import train_models, select_best_model, evaluate_model, save_model

log = logging.getLogger(__name__)

def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    xgb_params: dict = None,
    rf_params: dict = None
) -> dict:
    """
    Train and evaluate both XGBoost and Random Forest models.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        y_train: Training labels
        y_val: Validation labels
        y_test: Test labels
        xgb_params: XGBoost parameters
        rf_params: Random Forest parameters
    
    Returns:
        Dictionary containing models, evaluation results, and best model info
    """
    log.info("Starting model training and evaluation...")
    
    if xgb_params is None:
        xgb_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'mlogloss'
        }
    
    if rf_params is None:
        rf_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
    
    mlflow.autolog(
        log_model_signatures=True,
        log_input_examples=True,
        silent=True,
        log_datasets=False
    )
    
    log_dataset_info(X_train, "train", "Training dataset")
    log_dataset_info(X_val, "validation", "Validation dataset") 
    log_dataset_info(X_test, "test", "Test dataset")
    
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("validation_samples", len(X_val))
    mlflow.log_param("test_samples", len(X_test))
    mlflow.log_param("train_features", len(X_train.columns))
    mlflow.log_param("validation_features", len(X_val.columns))
    mlflow.log_param("test_features", len(X_test.columns))
    
    models_dict = train_models(X_train, y_train, X_val, y_val, xgb_params, rf_params)
    
    best_model_name, best_model = select_best_model(models_dict, X_val, y_val)
    
    mlflow.log_param("best_model", best_model_name)
    
    test_metrics, test_report = evaluate_model(best_model, X_test, y_test)
    
    for metric_name, metric_value in test_metrics.items():
        mlflow.log_metric(f"test_{metric_name}", metric_value)
    
    test_report_df = pd.DataFrame(test_report).transpose()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        test_report_df.to_csv(tmp_file.name, index=True)
        temp_path = tmp_file.name
    mlflow.log_artifact(temp_path, "test_classification_report.csv")
    os.unlink(temp_path)  
    
    if best_model_name == "xgboost":
        mlflow.xgboost.log_model(best_model, "best_model")
    else:
        mlflow.sklearn.log_model(best_model, "best_model")
    
    log.info(f"Model training and evaluation completed. Best model: {best_model_name}")
    log.info(f"Test F1 Score: {test_metrics['f1_score']:.4f}")
    
    return (
        models_dict,
        best_model_name,
        best_model,
        test_metrics,
        test_report,
        {
            'xgboost': models_dict['xgboost']['feature_importance'],
            'random_forest': models_dict['random_forest']['feature_importance']
        }
    )

def save_trained_models(
    models_dict: dict,
    best_model_name: str,
    model_save_path: str
) -> str:
    """
    Save trained models to disk.
    
    Args:
        models_dict: Dictionary containing trained models
        best_model_name: Name of the best model
        model_save_path: Path to save the models
    
    Returns:
        Path to the saved best model
    """
    log.info("Saving trained models...")
    
    os.makedirs(model_save_path, exist_ok=True)
    
    for model_name, model_info in models_dict.items():
        model_path = os.path.join(model_save_path, f"{model_name}_model.pkl")
        save_model(model_info['model'], model_name, model_path)
        
        importance_path = os.path.join(model_save_path, f"{model_name}_feature_importance.csv")
        model_info['feature_importance'].to_csv(importance_path, index=False)
    
    best_model = models_dict[best_model_name]['model']
    best_model_path = os.path.join(model_save_path, "best_model")
    
    if os.path.exists(best_model_path):
        shutil.rmtree(best_model_path)
    
    if best_model_name == "xgboost":
        mlflow.xgboost.save_model(best_model, best_model_path)
    else:
        mlflow.sklearn.save_model(best_model, best_model_path)
    
    log.info(f"Models saved to {model_save_path}")
    log.info(f"Best model ({best_model_name}) saved to {best_model_path}")
    
    return best_model_path

def generate_predictions(
    best_model,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> tuple:
    """
    Generate predictions using the best model.
    
    Args:
        best_model: Trained best model
        X_test: Test features
        y_test: Test labels (DataFrame)
    
    Returns:
        Tuple containing predictions, probabilities, results DataFrame, and confusion matrix
    """
    log.info("Generating predictions...")
    
    # Handle y_test input - convert to series if needed
    if isinstance(y_test, pd.DataFrame):
        y_test_series = y_test.iloc[:, 0]  # Take first column
    else:
        y_test_series = y_test
    
    mlflow.autolog(
        log_model_signatures=True,
        log_input_examples=True,
        silent=True,
        log_datasets=False
    )
    
    log_dataset_info(X_test, "test_predictions", "Test dataset for predictions")
    
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    results_df = pd.DataFrame({
        'true_label': y_test_series.values,
        'predicted_label': y_pred,
        'prediction_confidence': np.max(y_pred_proba, axis=1)
    })
    
    for i in range(y_pred_proba.shape[1]):
        results_df[f'prob_class_{i}'] = y_pred_proba[:, i]
    
    cm = confusion_matrix(y_test_series, y_pred)
    cm_df = pd.DataFrame(cm, 
                        index=[f'True_{i}' for i in range(len(cm))],
                        columns=[f'Pred_{i}' for i in range(len(cm))])
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        cm_df.to_csv(tmp_file.name, index=True)
        temp_path = tmp_file.name
    mlflow.log_artifact(temp_path, "confusion_matrix.csv")
    os.unlink(temp_path)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        results_df.to_csv(tmp_file.name, index=False)
        temp_path = tmp_file.name
    mlflow.log_artifact(temp_path, "detailed_predictions.csv")
    os.unlink(temp_path)
    
    mlflow.log_metric("total_predictions", len(y_pred))
    mlflow.log_metric("unique_predictions", len(np.unique(y_pred)))
    mlflow.log_metric("avg_confidence", results_df['prediction_confidence'].mean())
    mlflow.log_metric("min_confidence", results_df['prediction_confidence'].min())
    mlflow.log_metric("max_confidence", results_df['prediction_confidence'].max())
    
    log.info("Predictions generated and saved")
    
    return y_pred, y_pred_proba, results_df, cm 