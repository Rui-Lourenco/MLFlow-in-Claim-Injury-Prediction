import logging
import pandas as pd
from typing import Dict, Any

log = logging.getLogger(__name__)

def prepare_data_for_splitting(
    data: pd.DataFrame,
    target_column: str
) -> pd.DataFrame:
    """
    Prepare data for splitting by ensuring it has the correct structure.
    
    Args:
        data: Input dataframe
        target_column: Name of the target column
        
    Returns:
        Prepared dataframe ready for splitting
    """
    
    log.info(f"Preparing data for splitting. Shape: {data.shape}")
    
    # Make a copy to avoid modifying the original
    prepared_data = data.copy()
    
    # Ensure target column exists
    if target_column not in prepared_data.columns:
        log.warning(f"Target column '{target_column}' not found in data. Available columns: {list(prepared_data.columns)}")
        # If target column doesn't exist, we'll need to handle this case
        # For now, let's assume it exists or create a placeholder
        if "Claim Injury Type" in prepared_data.columns:
            log.info("Using 'Claim Injury Type' as target column")
            target_column = "Claim Injury Type"
        else:
            log.error("No suitable target column found")
            raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Remove any completely empty rows
    prepared_data = prepared_data.dropna(how='all')
    
    # Remove any completely empty columns
    prepared_data = prepared_data.dropna(axis=1, how='all')
    
    # Reset index to ensure clean indexing
    prepared_data = prepared_data.reset_index(drop=True)
    
    log.info(f"Data prepared successfully. Final shape: {prepared_data.shape}")
    log.info(f"Target column: {target_column}")
    log.info(f"Available columns: {list(prepared_data.columns)}")
    
    return prepared_data 