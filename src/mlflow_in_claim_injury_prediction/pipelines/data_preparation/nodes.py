import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import mlflow
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.utils import apply_one_hot_encoding, sine_cosine_encoding, extract_dates_components
from src.mlflow_in_claim_injury_prediction.utils.mlflow_utils import (
    log_dataset_info, create_experiment_run_name
)

log = logging.getLogger(__name__)

def encode_data(
    data: pd.DataFrame,
    target_column: str,
    date_columns: list = None
) -> pd.DataFrame:
    """
    Encode data with all pre-split encodings: target encoding, one-hot encoding, 
    sine-cosine encoding, and date component extraction. Also drop columns as in the notebook.
    
    Args:
        data: Input dataframe
        target_column: Name of the target column
        date_columns: List of date column names for component extraction
        
    Returns:
        Encoded dataframe ready for splitting
    """
    log.info(f"Input data shape: {data.shape}")
    
    # Create copy to avoid modifying original
    encoded_data = data.copy()
    
    # Step 1: Data Cleaning
    encoded_data = _clean_data(encoded_data)
    
    # Step 2: Find and validate target column
    target_column = _find_target_column(encoded_data, target_column)
    
    # Step 3: Encode target column
    encoded_data = _encode_target(encoded_data, target_column)
    
    # Step 4: Extract date components
    if date_columns:
        encoded_data = _extract_date_components(encoded_data, date_columns)
    
    # Step 5: Apply one-hot encoding
    encoded_data = _apply_one_hot_encoding(encoded_data)
    
    # Step 6: Apply sine-cosine encoding
    encoded_data = _apply_sine_cosine_encoding(encoded_data)
    
    # Step 7: Drop columns as in the notebook (including frequency encoding columns and target)
    columns_to_drop = [
        "Attorney/Representative", "Carrier Type", "COVID-19 Indicator", "Gender", "Medical Fee Region",
        "Carrier Name", "First Hearing Date", "C-2 Date", "C-3 Date", "Assembly Date",
        "Industry Code Description", "WCIO Cause of Injury Description", "WCIO Nature of Injury Description",
        "WCIO Part Of Body Description", "Agreement Reached", "WCB Decision",
        # Add frequency encoding columns to be dropped
        "County of Injury", "District Name", "Zip Code", "Industry Code",
        "WCIO Cause of Injury Code", "WCIO Nature of Injury Code", "WCIO Part Of Body Code"
    ]
    # Do NOT drop the target column
    columns_to_drop = [col for col in columns_to_drop if col != target_column]
    available_to_drop = [col for col in columns_to_drop if col in encoded_data.columns]
    if available_to_drop:
        log.info(f"Dropping columns before splitting: {available_to_drop}")
        encoded_data = encoded_data.drop(columns=available_to_drop)
    else:
        log.info("No columns to drop before splitting.")
    
    # Log final results
    _log_encoding_results(encoded_data, data.shape[1])
    
    return encoded_data

def _clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean data by removing empty rows/columns and duplicates."""
    log.info("Cleaning data...")
    
    # Remove completely empty rows and columns
    data = data.dropna(how='all').dropna(axis=1, how='all')
    
    # Remove duplicates
    original_shape = data.shape
    data = data.drop_duplicates()
    removed_duplicates = original_shape[0] - data.shape[0]
    
    if removed_duplicates > 0:
        log.info(f"Removed {removed_duplicates} duplicate rows")
        mlflow.log_metric("removed_duplicates", removed_duplicates)
    
    # Reset index
    data = data.reset_index(drop=True)
    
    log.info(f"Data cleaned. Shape: {data.shape}")
    return data

def _find_target_column(data: pd.DataFrame, target_column: str) -> str:
    """Find the target column with multiple name variations."""
    possible_names = [
        target_column,
        target_column.lower(),
        target_column.upper(),
        target_column.replace('_', ' '),
        target_column.replace(' ', '_'),
        'claim_injury_type',
        'Claim Injury Type',
        'CLAIM_INJURY_TYPE'
    ]
    
    for col_name in possible_names:
        if col_name in data.columns:
            log.info(f"Found target column: '{col_name}'")
            return col_name
    
    log.error(f"Target column not found. Tried: {possible_names}")
    log.error(f"Available columns: {list(data.columns)}")
    raise ValueError(f"Target column '{target_column}' not found in data")

def _encode_target(data: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Encode the target column using LabelEncoder."""
    log.info("Encoding target column...")
    
    # Check for NaN values in target
    target_nan_count = data[target_column].isnull().sum()
    if target_nan_count > 0:
        log.warning(f"Found {target_nan_count} NaN values in target column")
        mlflow.log_metric("target_nan_count", target_nan_count)
    
    # Encode target
    label_encoder = LabelEncoder()
    data[target_column] = label_encoder.fit_transform(data[target_column])
    
    # Log encoding information
    original_classes = label_encoder.classes_
    encoding_mapping = dict(zip(original_classes, range(len(original_classes))))
    
    log.info(f"Target encoding mapping: {encoding_mapping}")
    mlflow.log_param("target_classes", list(original_classes))
    mlflow.log_param("target_encoding_mapping", encoding_mapping)
    mlflow.log_metric("target_classes_count", len(original_classes))
    
    return data

def _extract_date_components(data: pd.DataFrame, date_columns: list) -> pd.DataFrame:
    """Extract date components from date columns."""
    log.info("Extracting date components...")
    
    # Find available date columns
    available_date_columns = [col for col in date_columns if col in data.columns]
    
    if available_date_columns:
        data = extract_dates_components([data], available_date_columns)[0]
        log.info(f"Extracted date components from {len(available_date_columns)} date columns")
        mlflow.log_metric("date_components_extracted", len(available_date_columns))
    else:
        log.warning("No date columns found for component extraction")
    
    return data

def _apply_one_hot_encoding(data: pd.DataFrame) -> pd.DataFrame:
    """Apply one-hot encoding to categorical features."""
    log.info("Applying one-hot encoding...")
    
    # Features for one-hot encoding (from notebooks)
    one_hot_features = [
        "Alternative Dispute Resolution",
        "Attorney/Representative", 
        "Carrier Type",
        "COVID-19 Indicator",
        "Gender",
        "Medical Fee Region"
    ]
    
    # Filter to available features
    available_features = [col for col in one_hot_features if col in data.columns]
    
    if available_features:
        # Create dummy dataframe for one-hot encoding
        dummy_df = data.copy()
        data, _ = apply_one_hot_encoding(data, dummy_df, available_features, save_encoder=False)
        log.info(f"Applied one-hot encoding to {len(available_features)} features")
        mlflow.log_metric("one_hot_encoded_features", len(available_features))
    else:
        log.warning("No one-hot encoding features found in the data")
    
    return data

def _apply_sine_cosine_encoding(data: pd.DataFrame) -> pd.DataFrame:
    """Apply sine-cosine encoding to cyclic features."""
    log.info("Applying sine-cosine encoding...")
    
    if 'Accident_Season' in data.columns:
        season_mapping = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}
        data = sine_cosine_encoding(data, "Accident_Season", season_mapping)
        log.info("Applied sine-cosine encoding to Accident_Season")
        mlflow.log_metric("sine_cosine_encoded_features", 1)
    else:
        log.warning("Accident_Season column not found for sine-cosine encoding")
    
    return data

def _log_encoding_results(data: pd.DataFrame, original_features: int):
    """Log final encoding results."""
    final_features = data.shape[1]
    new_features = final_features - original_features
    
    log.info(f"Encoding completed successfully")
    log.info(f"Original features: {original_features}")
    log.info(f"Final features: {final_features}")
    log.info(f"New features created: {new_features}")
    
    mlflow.log_metric("original_features", original_features)
    mlflow.log_metric("final_features", final_features)
    mlflow.log_metric("new_features_created", new_features)
    
    log_dataset_info(data, "data_after_encoding", "Data after encoding") 