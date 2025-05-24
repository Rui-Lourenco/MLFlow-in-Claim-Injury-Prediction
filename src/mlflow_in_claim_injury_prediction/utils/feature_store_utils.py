
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

def create_single_feature_group(
    feature_store,
    data: pd.DataFrame,
    group_name: str,
    feature_group_version: int,
    description: str,
    group_description: Dict[str, str],
    suite_name: Optional[str] = None,
    gx_context: Optional[gx.DataContext] = None
):
    """
    Create a single feature group using your existing GX setup.
    
    Args:
        feature_store: Hopsworks feature store object
        data: DataFrame with features
        group_name: Name of the feature group
        feature_group_version: Version number
        description: Description of the feature group
        group_description: Dictionary of feature descriptions
        suite_name: Name of expectation suite (optional)
        gx_context: Great Expectations context (optional)
    """
    
    logger.info(f"Creating feature group: {group_name}")
    
    # Get expectation suite if specified
    expectation_suite = None
    if suite_name:
        try:
            expectation_suite = get_or_create_expectation_suite(suite_name, gx_context)
            # Validate data before creating feature group
            is_valid = validate_data_with_gx(data, suite_name, gx_context)
            if not is_valid:
                logger.warning(f"Data validation failed for {group_name}, but proceeding...")
        except Exception as e:
            logger.warning(f"Could not use expectation suite {suite_name}: {e}")
    
    # Create feature group
    object_feature_group = feature_store.get_or_create_feature_group(
        name=group_name,
        version=feature_group_version,
        description=description,
        primary_key=["index"],
        event_time="datetime",
        online_enabled=False,
        expectation_suite=expectation_suite,
    )
    
    try:
        # Upload the data to the feature group
        object_feature_group.insert(
            features=data,
            overwrite=False,
            storage="offline",
            write_options={"wait_for_job": True},
        )
        
        # Add feature descriptions
        for feature_name, feature_desc in group_description.items():
            if feature_name in data.columns:
                object_feature_group.update_feature_description(feature_name, feature_desc)
        
        # Configure and update statistics
        object_feature_group.statistics_config = {
            "enabled": True,
            "histograms": True,
            "correlations": True,
        }
        object_feature_group.update_statistics_config()
        object_feature_group.compute_statistics()
        
        logger.info(f"Successfully created feature group: {group_name}")
        return object_feature_group
        
    except Exception as e:
        logger.error(f"Error during feature group creation or data insertion for {group_name}: {e}")
        raise
