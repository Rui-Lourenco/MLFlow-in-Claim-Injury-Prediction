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
from src.mlflow_in_claim_injury_prediction.utils.mlflow_utils import (
    setup_mlflow, start_mlflow_run, log_dataset_info, log_model_with_metadata,
    log_feature_importance, log_predictions_with_metadata, log_model_evaluation_metrics,
    create_experiment_run_name, log_dataframe_as_artifact
)

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
    
    # Setup MLflow
    setup_mlflow()
    
    # Create descriptive run name
    run_name = create_experiment_run_name("complete_model_training")
    
    with start_mlflow_run(run_name=run_name, tags={"pipeline": "model_inference"}) as run:
        log.info(f"Starting complete model training run: {run.info.run_id}")
        
        # Log dataset information
        log_dataset_info(X_train, "train_features", "Training features")
        log_dataset_info(X_val, "val_features", "Validation features")
        log_dataset_info(X_test, "test_features", "Test features")
        
        # Log target distributions
        train_target_dist = y_train.value_counts()
        val_target_dist = y_val.value_counts()
        test_target_dist = y_test.value_counts()
        
        log_dataframe_as_artifact(train_target_dist.to_frame(), "train_target_distribution.csv")
        log_dataframe_as_artifact(val_target_dist.to_frame(), "val_target_distribution.csv")
        log_dataframe_as_artifact(test_target_dist.to_frame(), "test_target_distribution.csv")
        
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
        
        # Log best model selection
        mlflow.log_param("best_model", best_model_name)
        
        # Evaluate best model on test set
        test_metrics, test_report = evaluate_model(best_model, X_test, y_test)
        
        # Log test metrics
        log_model_evaluation_metrics(test_metrics, "test")
        
        # Log test report
        log_dataframe_as_artifact(pd.DataFrame(test_report), "test_classification_report.csv")
        
        # Log best model with metadata
        best_model_params = xgb_params if best_model_name == "xgboost" else rf_params
        log_model_with_metadata(
            model=best_model,
            model_name=f"best_{best_model_name}",
            model_type=best_model_name,
            feature_names=X_test.columns.tolist(),
            model_params=best_model_params
        )
        
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
    
    # Setup MLflow for prediction logging
    setup_mlflow()
    
    # Create descriptive run name
    run_name = create_experiment_run_name("model_predictions")
    
    with start_mlflow_run(run_name=run_name, tags={"pipeline": "predictions"}) as run:
        log.info(f"Starting prediction generation run: {run.info.run_id}")
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)
        
        # Log predictions with metadata
        log_predictions_with_metadata(y_test.values, y_pred, y_pred_proba, "test")
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'true_label': y_test,
            'predicted_label': y_pred,
            'prediction_confidence': np.max(y_pred_proba, axis=1)
        })
        
        # Add probability columns for each class
        for i in range(y_pred_proba.shape[1]):
            results_df[f'prob_class_{i}'] = y_pred_proba[:, i]
        
        # Calculate additional metrics
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, 
                            index=[f'True_{i}' for i in range(len(cm))],
                            columns=[f'Pred_{i}' for i in range(len(cm))])
        
        # Log confusion matrix
        log_dataframe_as_artifact(cm_df, "confusion_matrix.csv")
        
        # Log detailed results
        log_dataframe_as_artifact(results_df, "detailed_predictions.csv")
        
        # Log prediction statistics
        mlflow.log_metric("total_predictions", len(y_pred))
        mlflow.log_metric("unique_predictions", len(np.unique(y_pred)))
        mlflow.log_metric("avg_confidence", results_df['prediction_confidence'].mean())
        mlflow.log_metric("min_confidence", results_df['prediction_confidence'].min())
        mlflow.log_metric("max_confidence", results_df['prediction_confidence'].max())
        
        log.info("Predictions generated and saved")
        
        return {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'results_df': results_df,
            'confusion_matrix': cm
        } 