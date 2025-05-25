
import pandas as pd
from typing import Dict, Any
import logging
import great_expectations as gx
import time
import hopsworks
from ...utils.feature_store_utils import (
    connect_to_feature_store, 
    get_gx_context,
    validate_data_with_gx
)

logger = logging.getLogger(__name__)

def create_feature_groups_with_gx_robust(
    data: pd.DataFrame,
    feature_store_params: Dict[str, Any],
    feature_groups_config: Dict[str, Any],
    use_expectations: bool = True
) -> Dict[str, Any]:
    """
    Create feature groups using Great Expectations with robust error handling.
    
    Args:
        data: Input dataframe with all features
        feature_store_params: Feature store connection parameters
        feature_groups_config: Configuration for feature groups
        use_expectations: Whether to use GX validation (default: True)
    
    Returns:
        Dict with metadata about created feature groups
    """
    
    logger.info("=== Robust Feature Groups Creation with Great Expectations ===")
    logger.info(f"Data shape: {data.shape}")
    
    # Get GX context if using expectations
    gx_context = None
    if use_expectations:
        try:
            gx_context = get_gx_context()
            logger.info("✅ Successfully connected to existing Great Expectations setup")
        except Exception as e:
            logger.warning(f"⚠️  Could not connect to GX: {e}. Proceeding without validation.")
            use_expectations = False
    
    # Connect to feature store
    try:
        feature_store = connect_to_feature_store(
            api_key=feature_store_params["api_key"],
            project_name=feature_store_params["project_name"]
        )
    except Exception as e:
        logger.error(f"❌ Failed to connect to feature store: {e}")
        return {"error": str(e)}
    
    # Check existing feature groups to avoid duplicates
    # Check existing feature groups (with API fix)
    existing_feature_groups = []
    try:
        try:
            existing_fgs = feature_store.get_feature_groups()
        except TypeError:
            existing_fgs = []
            logger.info("📋 Could not list existing feature groups - will create all")
        
        existing_feature_groups = [fg.name for fg in existing_fgs]
        logger.info(f"📋 Found {len(existing_feature_groups)} existing feature groups")
    except Exception as e:
        logger.warning(f"⚠️  Could not check existing feature groups: {e}")
        existing_feature_groups = []
    
    created_feature_groups = {}
    failed_feature_groups = {}
    skipped_feature_groups = {}
    base_version = feature_store_params.get("base_version", 1)
    
    # Add required columns if missing
    if "index" not in data.columns:
        data = data.reset_index()
    if "datetime" not in data.columns:
        data["datetime"] = pd.Timestamp.now()
    
    for group_name, config in feature_groups_config.items():
        
        # Skip if already exists
        if group_name in existing_feature_groups:
            logger.info(f"⏭️  Skipping {group_name} - already exists")
            skipped_feature_groups[group_name] = "Already exists"
            continue
        
        # Check available columns
        available_columns = [col for col in config["columns"] if col in data.columns]
        missing_columns = [col for col in config["columns"] if col not in data.columns]
        
        if not available_columns:
            logger.warning(f"⏭️  Skipping {group_name} - no columns available")
            skipped_feature_groups[group_name] = f"No columns available: {config['columns']}"
            continue
        
        # Create feature group with robust retry logic
        try:
            result = create_feature_group_with_gx_and_retry(
                feature_store=feature_store,
                data=data,
                group_name=group_name,
                config=config,
                available_columns=available_columns,
                missing_columns=missing_columns,
                base_version=base_version,
                use_expectations=use_expectations,
                gx_context=gx_context
            )
            
            if result["success"]:
                created_feature_groups[group_name] = result["metadata"]
                logger.info(f"✅ Successfully created: {group_name}")
            else:
                failed_feature_groups[group_name] = result["error"]
                logger.error(f"❌ Failed to create: {group_name} - {result['error']}")
        
        except Exception as e:
            logger.error(f"❌ Unexpected error for {group_name}: {e}")
            failed_feature_groups[group_name] = str(e)
            continue
    
    result = {
        "created_feature_groups": created_feature_groups,
        "failed_feature_groups": failed_feature_groups,
        "skipped_feature_groups": skipped_feature_groups,
        "total_groups_created": len(created_feature_groups),
        "total_failed": len(failed_feature_groups),
        "total_skipped": len(skipped_feature_groups),
        "used_great_expectations": use_expectations,
        "data_shape": data.shape
    }
    
    logger.info(f"=== SUMMARY ===")
    logger.info(f"✅ Created: {len(created_feature_groups)} feature groups")
    logger.info(f"❌ Failed: {len(failed_feature_groups)} feature groups")
    logger.info(f"⏭️  Skipped: {len(skipped_feature_groups)} feature groups")
    logger.info(f"🔍 Used Great Expectations: {use_expectations}")
    
    return result

def create_feature_group_with_gx_and_retry(
    feature_store,
    data: pd.DataFrame,
    group_name: str,
    config: Dict,
    available_columns: list,
    missing_columns: list,
    base_version: int,
    use_expectations: bool = True,
    gx_context = None,
    max_retries: int = 3,
    chunk_size: int = 50000
) -> Dict[str, Any]:
    """
    Create feature group with Great Expectations validation and robust retry logic.
    """
    
    logger.info(f"🔄 Creating feature group: {group_name}")
    logger.info(f"   Available columns: {available_columns}")
    if missing_columns:
        logger.warning(f"   Missing columns: {missing_columns}")
    
    # Prepare data for this feature group
    group_data_columns = ["index", "datetime"] + available_columns
    group_data = data[group_data_columns].copy()
    
    logger.info(f"   Data shape: {group_data.shape}")
    
    # Validate with Great Expectations if enabled
    if use_expectations and gx_context:
        suite_name = f"{group_name}_expectations"
        try:
            logger.info(f"   🔍 Validating data with Great Expectations...")
            is_valid = validate_data_with_gx(group_data, suite_name, gx_context)
            if not is_valid:
                logger.warning(f"   ⚠️  Data validation failed for {group_name}, but proceeding...")
        except Exception as e:
            logger.warning(f"   ⚠️  Could not validate with GX: {e}")
    
    # Filter descriptions for available columns only
    available_descriptions = {
        k: v for k, v in config["features_desc"].items() 
        if k in available_columns
    }
    
    for attempt in range(max_retries):
        try:
            logger.info(f"   📤 Attempt {attempt + 1}/{max_retries}")
            
            # Create feature group object
            feature_group = feature_store.get_or_create_feature_group(
                name=group_name,
                version=base_version,
                description=config["description"],
                primary_key=["index"],
                event_time="datetime",
                online_enabled=False,
            )
            
            # Upload data with chunking for large datasets
            upload_success = upload_data_smart(feature_group, group_data, chunk_size)
            
            if not upload_success:
                raise Exception("Data upload failed")
            
            # Add feature descriptions (with sanitized names)
            try:
                add_feature_descriptions_robust(feature_group, available_descriptions, available_columns)
            except Exception as e:
                logger.warning(f"   ⚠️  Could not add descriptions: {e}")
            
            # Configure statistics
            try:
                feature_group.statistics_config = {
                    "enabled": True,
                    "histograms": True,
                    "correlations": True,
                }
                feature_group.update_statistics_config()
                feature_group.compute_statistics()
                logger.info(f"   📊 Statistics configured successfully")
            except Exception as e:
                logger.warning(f"   ⚠️  Could not configure statistics: {e}")
            
            return {
                "success": True,
                "metadata": {
                    "name": feature_group.name,
                    "version": feature_group.version,
                    "columns": available_columns,
                    "missing_columns": missing_columns,
                    "row_count": len(group_data),
                    "attempt": attempt + 1,
                    "expectation_suite": f"{group_name}_expectations" if use_expectations else None
                }
            }
            
        except Exception as e:
            logger.warning(f"   ⚠️  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30  # Exponential backoff: 30s, 60s, 90s
                logger.info(f"   ⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                return {
                    "success": False,
                    "error": f"All {max_retries} attempts failed. Last error: {str(e)}"
                }

def upload_data_smart(feature_group, data: pd.DataFrame, chunk_size: int) -> bool:
    """Smart data upload - chunked for large datasets, direct for small ones."""
    
    if len(data) > chunk_size:
        logger.info(f"   📦 Large dataset detected. Uploading in chunks of {chunk_size:,}")
        return upload_data_in_chunks(feature_group, data, chunk_size)
    else:
        logger.info(f"   📤 Small dataset. Uploading directly ({len(data):,} rows)")
        return upload_data_direct(feature_group, data)

def upload_data_in_chunks(feature_group, data: pd.DataFrame, chunk_size: int) -> bool:
    """Upload data in chunks to avoid Kafka timeouts."""
    
    total_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size != 0 else 0)
    logger.info(f"   📦 Uploading {len(data):,} rows in {total_chunks} chunks")
    
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i + chunk_size].copy()
        chunk_num = (i // chunk_size) + 1
        
        logger.info(f"   📤 Uploading chunk {chunk_num}/{total_chunks} ({len(chunk):,} rows)")
        
        try:
            feature_group.insert(
                features=chunk,
                overwrite=False,
                storage="offline",
                write_options={
                    "wait_for_job": False,  # Don't wait for each chunk
                },
            )
            
            # Small delay between chunks to avoid overwhelming Kafka
            time.sleep(5)  # Increased delay for more stability
            
        except Exception as e:
            logger.error(f"   ❌ Failed to upload chunk {chunk_num}: {e}")
            return False
    
    logger.info("   ✅ All chunks uploaded successfully")
    return True

def upload_data_direct(feature_group, data: pd.DataFrame) -> bool:
    """Upload data directly for smaller datasets."""
    
    try:
        feature_group.insert(
            features=data,
            overwrite=False,
            storage="offline",
            write_options={
                "wait_for_job": True,
            },
        )
        logger.info("   ✅ Direct upload successful")
        return True
    except Exception as e:
        logger.error(f"   ❌ Direct upload failed: {e}")
        return False

def add_feature_descriptions_robust(feature_group, features_desc: dict, available_columns: list):
    """Add feature descriptions with name sanitization and error handling."""
    
    def sanitize_feature_name(name: str) -> str:
        """Convert feature name to Hopsworks format."""
        return name.lower().replace(' ', '_').replace('-', '_')
    
    logger.info(f"   📝 Adding descriptions for {len(features_desc)} features")
    
    for feature_name, feature_desc in features_desc.items():
        if feature_name in available_columns:
            sanitized_name = sanitize_feature_name(feature_name)
            try:
                feature_group.update_feature_description(sanitized_name, feature_desc)
                logger.debug(f"   ✅ Added description for: {sanitized_name}")
            except Exception as e:
                logger.warning(f"   ⚠️  Could not update description for {sanitized_name}: {e}")