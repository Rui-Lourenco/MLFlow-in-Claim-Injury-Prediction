
import pandas as pd
from typing import Dict, Any
import logging
import great_expectations as gx
from ...utils.feature_store_utils import (
    connect_to_feature_store, 
    create_single_feature_group,
    get_gx_context,
    validate_data_with_gx
)

logger = logging.getLogger(__name__)

def create_feature_groups_with_gx(
    data: pd.DataFrame,
    feature_store_params: Dict[str, Any],
    feature_groups_config: Dict[str, Any],
    use_expectations: bool = True
) -> Dict[str, Any]:
    """
    Create feature groups using your existing Great Expectations setup.
    
    Args:
        data: Input dataframe with all features
        feature_store_params: Feature store connection parameters
        feature_groups_config: Configuration for feature groups
        use_expectations: Whether to use GX validation (default: True)
    
    Returns:
        Dict with metadata about created feature groups
    """
    
    # Get GX context if using expectations
    gx_context = None
    if use_expectations:
        try:
            gx_context = get_gx_context()
            logger.info("Successfully connected to existing Great Expectations setup")
        except Exception as e:
            logger.warning(f"Could not connect to GX: {e}. Proceeding without validation.")
            use_expectations = False
    
    # Connect to feature store
    feature_store = connect_to_feature_store(
        api_key=feature_store_params["api_key"],
        project_name=feature_store_params["project_name"]
    )
    
    created_feature_groups = {}
    base_version = feature_store_params.get("base_version", 1)
    
    # Add required columns if missing
    if "index" not in data.columns:
        data = data.reset_index()
    if "datetime" not in data.columns:
        data["datetime"] = pd.Timestamp.now()
    
    for group_name, config in feature_groups_config.items():
        try:
            # Filter data for this feature group
            available_columns = [col for col in config["columns"] if col in data.columns]
            
            if not available_columns:
                logger.warning(f"No columns found for {group_name}. Skipping...")
                continue
            
            # Prepare data for this feature group
            group_data_columns = ["index", "datetime"] + available_columns
            group_data = data[group_data_columns].copy()
            
            # Filter descriptions for available columns only
            available_descriptions = {
                k: v for k, v in config["features_desc"].items() 
                if k in available_columns
            }
            
            # Define expectation suite name (you can customize this)
            suite_name = f"{group_name}_expectations" if use_expectations else None
            
            # Create the feature group
            feature_group = create_single_feature_group(
                feature_store=feature_store,
                data=group_data,
                group_name=group_name,
                feature_group_version=base_version,
                description=config["description"],
                group_description=available_descriptions,
                suite_name=suite_name,
                gx_context=gx_context
            )
            
            created_feature_groups[group_name] = {
                "name": feature_group.name,
                "version": feature_group.version,
                "columns": available_columns,
                "description": config["description"],
                "expectation_suite": suite_name
            }
            
        except Exception as e:
            logger.error(f"Failed to create feature group {group_name}: {e}")
            continue
    
    return {
        "created_feature_groups": created_feature_groups,
        "total_groups_created": len(created_feature_groups),
        "used_great_expectations": use_expectations
    }
