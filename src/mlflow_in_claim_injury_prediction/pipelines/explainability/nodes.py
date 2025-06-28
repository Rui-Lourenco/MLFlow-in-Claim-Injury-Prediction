import logging
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance

import mlflow

from src.mlflow_in_claim_injury_prediction.utils.mlflow_utils import (
    create_experiment_run_name, log_dataframe_as_artifact
)

log = logging.getLogger(__name__)

def calculate_permutation_importance(
    model: BaseEstimator, 
    X: pd.DataFrame, 
    y: pd.Series,
    n_repeats: int = None, 
    random_state: int = None
) -> pd.DataFrame:
    """
    Calculate permutation importance for a given model.
    
    Args:
        model: The trained model.
        X: The feature matrix.
        y: The target vector.
        n_repeats: Number of times to permute a feature.
        random_state: Random state for reproducibility.
        
    Returns:
        DataFrame containing permutation importance scores.
    """
    
    run_name = create_experiment_run_name("explainability", "permutation_importance")
    
    with mlflow.start_run(run_name=run_name, nested=True):        
        log.info("Calculating permutation importance...")
        
        result = permutation_importance(
            model,
            X, y,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values(
            by='importance_mean',
            ascending=False
        )
        
        mlflow.log_params({
            'n_repeats': n_repeats,
            'random_state': random_state,
            'total_features': len(X.columns)
        })
        
        for i, row in importance_df.head(10).iterrows():
            mlflow.log_param(f"top_{i+1}_{row['feature']}_importance_mean", row['importance_mean'])
            mlflow.log_param(f"top_{i+1}_{row['feature']}_importance_std", row['importance_std'])
        
        mlflow.log_metric("top_feature_importance", importance_df.iloc[0]['importance_mean'])
        mlflow.log_metric("avg_feature_importance", importance_df['importance_mean'].mean())
        mlflow.log_metric("std_feature_importance", importance_df['importance_mean'].std())
        
        log_dataframe_as_artifact(importance_df, "permutation_importance_full.csv")
        log_dataframe_as_artifact(importance_df.head(20), "permutation_importance_top20.csv")
        
        log.info(f"Permutation importance calculated for {len(X.columns)} features")
        log.info(f"Top 5 most important features: {importance_df.head(5)['feature'].tolist()}")
            
    return importance_df 