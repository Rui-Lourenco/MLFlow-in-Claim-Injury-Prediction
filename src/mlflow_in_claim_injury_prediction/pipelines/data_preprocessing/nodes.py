import logging
import pandas as pd

log = logging.getLogger(__name__)

def preprocess_raw_data(
    raw_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Preprocess raw data to create processed data ready for further processing.
    
    Args:
        raw_data: Raw input dataframe
        
    Returns:
        Processed dataframe
    """
    
    log.info(f"Preprocessing raw data. Input shape: {raw_data.shape}")
    
    # Make a copy to avoid modifying the original
    processed_data = raw_data.copy()
    
    # Basic preprocessing steps
    # Remove any completely empty rows
    processed_data = processed_data.dropna(how='all')
    
    # Remove any completely empty columns
    processed_data = processed_data.dropna(axis=1, how='all')
    
    # Reset index to ensure clean indexing
    processed_data = processed_data.reset_index(drop=True)
    
    # Handle any obvious data type issues
    # Convert date columns to datetime 
    date_columns = [col for col in processed_data.columns if 'date' in col.lower()]
    for col in date_columns:
        try:
            processed_data[col] = pd.to_datetime(processed_data[col], errors='coerce')
        except:
            log.warning(f"Could not convert column {col} to datetime")
    
    # Convert numeric columns
    numeric_columns = [col for col in processed_data.columns if any(x in col.lower() for x in ['age', 'wage', 'count', 'number', 'days', 'year'])]
    for col in numeric_columns:
        try:
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
        except:
            log.warning(f"Could not convert column {col} to numeric")
    
    log.info(f"Data preprocessing completed. Output shape: {processed_data.shape}")
    log.info(f"Columns: {list(processed_data.columns)}")
    
    return processed_data 