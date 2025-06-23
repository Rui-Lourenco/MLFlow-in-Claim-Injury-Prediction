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
) -> None:
    """
    Upload data to the feature store.

    Args:
        df (pd.DataFrame): Data to upload.
        group_name (str): Name of the feature group.
        description (str): Description of the feature group.
        feature_descriptions (List[dict]): List of feature descriptions.
        suite_name (str): Name of the expectation suite.
    """
       
    df["datetime"] = datetime.now()
    
    if "index" not in df.columns:
        df = df.reset_index()
    
    df = df.applymap(lambda x: None if pd.isna(x) else x)
    
    settings_store = read_credentials()["SETTINGS_STORE"]
    suite = load_expectation_suite(suite_name)
    
    to_feature_store(
        data=df,
        group_name=group_name,
        feature_group_version=1,
        description=description if not None else "Data uploaded to the feature store",
        group_description=feature_descriptions,
        validation_expectation_suite=suite,
        SETTINGS=settings_store
    )
    
    log.info(f"Data uploaded to feature store: {group_name} | Shape: {df.shape}") 