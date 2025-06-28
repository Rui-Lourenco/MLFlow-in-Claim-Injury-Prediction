import pandas as pd
import hopsworks
import great_expectations as gx
from great_expectations.core.expectation_suite import ExpectationSuite
from typing import Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)

def get_gx_context():
    """Get the Great Expectations context from your existing setup."""
    try:
        # This will use your existing gx/ configuration
        context = gx.get_context()
        return context
    except Exception as e:
        logger.error(f"Failed to get GX context: {e}")
        raise

def get_or_create_expectation_suite(
    suite_name: str, 
    gx_context: Optional[gx.DataContext] = None
) -> ExpectationSuite:
    """
    Get existing expectation suite or create a new one using your GX setup.
    
    Args:
        suite_name: Name of the expectation suite
        gx_context: Great Expectations context (will get it if not provided)
    
    Returns:
        ExpectationSuite object
    """
    if gx_context is None:
        gx_context = get_gx_context()
    
    try:
        # Try to get existing suite first
        suite = gx_context.get_expectation_suite(suite_name)
        logger.info(f"Using existing expectation suite: {suite_name}")
        return suite
    except:
        # Create new suite if it doesn't exist
        logger.info(f"Creating new expectation suite: {suite_name}")
        suite = gx_context.create_expectation_suite(suite_name)
        return suite

def validate_data_with_gx(
    data: pd.DataFrame,
    suite_name: str,
    gx_context: Optional[gx.DataContext] = None
) -> bool:
    """
    Validate data using Great Expectations.
    
    Args:
        data: DataFrame to validate
        suite_name: Name of expectation suite to use
        gx_context: Great Expectations context
    
    Returns:
        True if validation passes, False otherwise
    """
    if gx_context is None:
        gx_context = get_gx_context()
    
    try:
        # Get the expectation suite
        suite = get_or_create_expectation_suite(suite_name, gx_context)
        
        # Create a validator
        validator = gx_context.get_validator(
            batch_request=gx.core.batch.RuntimeBatchRequest(
                datasource_name="pandas_datasource",  # You might need to adjust this
                data_connector_name="runtime_data_connector",
                data_asset_name="feature_store_data",
                runtime_parameters={"batch_data": data},
                batch_identifiers={"default_identifier_name": "feature_store_batch"}
            ),
            expectation_suite=suite
        )
        
        # Run validation
        validation_result = validator.validate()
        
        logger.info(f"Validation result for {suite_name}: {validation_result.success}")
        return validation_result.success
        
    except Exception as e:
        logger.error(f"Validation failed for {suite_name}: {e}")
        return False

def connect_to_feature_store(api_key: str, project_name: str):
    """Connect to Hopsworks feature store."""
    try:
        project = hopsworks.login(api_key_value=api_key, project=project_name)
        feature_store = project.get_feature_store()
        return feature_store
    except Exception as e:
        logger.error(f"Failed to connect to feature store: {e}")
        raise

