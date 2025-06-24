import logging
from typing import List
from datetime import datetime
import warnings; warnings.filterwarnings("ignore")

import pandas as pd
from .utils import to_feature_store, read_credentials, load_expectation_suite


log = logging.getLogger(__name__)

def upload_data(df: pd.DataFrame,
                group_name: str,
                description: str,
                feature_descriptions: List[dict],
                suite_name: str,
                feature_group_columns: List[str] = None
) -> None:
    """
    Upload data to the feature store.

    Args:
        df (pd.DataFrame): Data to upload.
        group_name (str): Name of the feature group.
        description (str): Description of the feature group.
        feature_descriptions (List[dict]): List of feature descriptions.
        suite_name (str): Name of the expectation suite.
        feature_group_columns (List[str], optional): List of columns expected for the feature group.
    """
    
    # Create a copy to avoid modifying the original
    upload_df = df.copy()
    
    # Filter columns to only those present in both DataFrame and config
    if feature_group_columns is not None:
        missing = [col for col in feature_group_columns if col not in upload_df.columns]
        if missing:
            log.warning(f"The following columns are expected by the feature group config but missing from the DataFrame: {missing}")
        present = [col for col in feature_group_columns if col in upload_df.columns]
        upload_df = upload_df[present]
    
    # Add timestamp column
    upload_df["datetime"] = datetime.now()
    
    # Reset index if needed
    if "index" not in upload_df.columns:
        upload_df = upload_df.reset_index()
    
    # Handle data types properly to avoid PyArrow errors
    for column in upload_df.columns:
        if upload_df[column].dtype == 'object':
            # Convert object columns to string, handling NaN values
            upload_df[column] = upload_df[column].astype('string').fillna('')
        elif upload_df[column].dtype == 'float64':
            # Keep float columns as is, but handle NaN values
            upload_df[column] = upload_df[column].fillna(0.0)
        elif upload_df[column].dtype == 'int64':
            # Keep int columns as is, but handle NaN values
            upload_df[column] = upload_df[column].fillna(0)
        elif 'datetime' in str(upload_df[column].dtype):
            # Keep datetime columns as is
            pass
    
    log.info(f"Prepared data for upload. Shape: {upload_df.shape}")
    log.info(f"Data types: {upload_df.dtypes.to_dict()}")
    
    settings_store = read_credentials()["SETTINGS_STORE"]
    suite = load_expectation_suite(suite_name)
    
    to_feature_store(
        data=upload_df,
        group_name=group_name,
        feature_group_version=1,
        description=description if not None else "Data uploaded to the feature store",
        group_description=feature_descriptions,
        validation_expectation_suite=suite,
        SETTINGS=settings_store
    )
    
    log.info(f"Data uploaded to feature store: {group_name} | Shape: {upload_df.shape}") 