import logging
import pandas as pd
import numpy as np
import mlflow
import sys
import os

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
    
    # Set default parameters if not provided
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
    
    # Log parameters
    mlflow.log_params({"xgb_" + k: v for k, v in xgb_params.items()})
    mlflow.log_params({"rf_" + k: v for k, v in rf_params.items()})
    
    # Train models
    models_dict = train_models(X_train, y_train, X_val, y_val, xgb_params, rf_params)
    
    # Select best model
    best_model_name, best_model = select_best_model(models_dict, X_val, y_val)
    
    # Evaluate best model on test set
    test_metrics, test_report = evaluate_model(best_model, X_test, y_test)
    
    # Log test metrics
    for metric_name, metric_value in test_metrics.items():
        mlflow.log_metric(f"test_{metric_name}", metric_value)
    
    # Log test report
    mlflow.log_artifact(pd.DataFrame(test_report).to_csv(), "test_classification_report.csv")
    
    # Prepare results
    results = {
        'models': models_dict,
        'best_model_name': best_model_name,
        'best_model': best_model,
        'test_metrics': test_metrics,
        'test_report': test_report,
        'feature_importance': {
            'xgboost': models_dict['xgboost']['feature_importance'],
            'random_forest': models_dict['random_forest']['feature_importance']
        }
    }
    
    log.info(f"Model training and evaluation completed. Best model: {best_model_name}")
    log.info(f"Test F1 Score: {test_metrics['f1_score']:.4f}")
    
    return results

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
    
    # Save all models
    for model_name, model_info in models_dict.items():
        model_path = os.path.join(model_save_path, f"{model_name}_model.pkl")
        save_model(model_info['model'], model_name, model_path)
        
        # Save feature importance
        importance_path = os.path.join(model_save_path, f"{model_name}_feature_importance.csv")
        model_info['feature_importance'].to_csv(importance_path, index=False)
    
    # Save best model separately
    best_model = models_dict[best_model_name]['model']
    best_model_path = os.path.join(model_save_path, "best_model.pkl")
    save_model(best_model, best_model_name, best_model_path)
    
    log.info(f"Models saved to {model_save_path}")
    log.info(f"Best model ({best_model_name}) saved to {best_model_path}")
    
    return best_model_path

def generate_predictions(
    best_model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Generate predictions using the best model.
    
    Args:
        best_model: Trained best model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary containing predictions and probabilities
    """
    log.info("Generating predictions...")
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred,
        'prediction_confidence': np.max(y_pred_proba, axis=1)
    })
    
    # Add probability columns for each class
    for i in range(y_pred_proba.shape[1]):
        results_df[f'prob_class_{i}'] = y_pred_proba[:, i]
    
    # Log predictions
    mlflow.log_artifact(results_df.to_csv(index=False), "test_predictions.csv")
    
    # Calculate additional metrics
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, 
                        index=[f'True_{i}' for i in range(len(cm))],
                        columns=[f'Pred_{i}' for i in range(len(cm))])
    
    mlflow.log_artifact(cm_df.to_csv(), "confusion_matrix.csv")
    
    log.info("Predictions generated and saved")
    
    return {
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'results_df': results_df,
        'confusion_matrix': cm
    } 