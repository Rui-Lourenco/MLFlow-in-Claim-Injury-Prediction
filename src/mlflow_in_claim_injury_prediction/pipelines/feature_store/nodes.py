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

def map_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """
    Map column names from original format to snake_case for feature store compatibility.
    
    Args:
        data: DataFrame with original column names
        
    Returns:
        DataFrame with snake_case column names
    """
    # Column name mapping from original to snake_case
    column_mapping = {
        # Personal Information
        "Age at Injury": "age_at_injury",
        "Birth Year": "birth_year",
        "Known Age at Injury": "known_age_at_injury",
        "Known Birth Year": "known_birth_year",
        
        # Gender (if exists)
        "Gender_M": "gender_m",
        "Gender_Unknown": "gender_unknown",
        
        # Financial Information
        "Average Weekly Wage": "average_weekly_wage",
        "Relative_Wage": "relative_wage",
        "Number of Dependents": "number_of_dependents",
        
        # Medical and Case Processing
        "IME-4 Count": "ime_4_count",
        "Attorney/Representative_Y": "attorney_representative_y",
        "Alternative Dispute Resolution_Y": "alternative_dispute_resolution_y",
        "Alternative Dispute Resolution_U": "alternative_dispute_resolution_u",
        
        # Time to Event Features
        "Days_to_First_Hearing": "days_to_first_hearing",
        "Days_to_C2": "days_to_c2",
        "Days_to_C3": "days_to_c3",
        
        # C-2 Date Components
        "C-2 Date_Year": "c_2_date_year",
        "C-2 Date_Month": "c_2_date_month",
        "C-2 Date_Day": "c_2_date_day",
        "C-2 Date_DayOfWeek": "c_2_date_dayofweek",
        "Known C-2 Date": "known_c_2_date",
        
        # C-3 Date Components
        "C-3 Date_Year": "c_3_date_year",
        "C-3 Date_Month": "c_3_date_month",
        "C-3 Date_Day": "c_3_date_day",
        "C-3 Date_DayOfWeek": "c_3_date_dayofweek",
        "Known C-3 Date": "known_c_3_date",
        
        # First Hearing Date Components
        "First Hearing Date_Year": "first_hearing_date_year",
        "First Hearing Date_Month": "first_hearing_date_month",
        "First Hearing Date_Day": "first_hearing_date_day",
        "First Hearing Date_DayOfWeek": "first_hearing_date_dayofweek",
        "Known First Hearing Date": "known_first_hearing_date",
        
        # Accident Date Components
        "Accident Date_Year": "accident_date_year",
        "Accident Date_Month": "accident_date_month",
        "Accident Date_Day": "accident_date_day",
        "Accident Date_DayOfWeek": "accident_date_dayofweek",
        "Known Accident Date": "known_accident_date",
        
        # Assembly Date Components
        "Assembly Date_Year": "assembly_date_year",
        "Assembly Date_Month": "assembly_date_month",
        "Assembly Date_Day": "assembly_date_day",
        "Assembly Date_DayOfWeek": "assembly_date_dayofweek",
        "Known Assembly Date": "known_assembly_date",
        
        # Carrier Information
        "Carrier Type_2A. SIF": "carrier_type_2a_sif",
        "Carrier Type_3A. SELF PUBLIC": "carrier_type_3a_self_public",
        "Carrier Type_4A. SELF PRIVATE": "carrier_type_4a_self_private",
        "Carrier Type_5D. SPECIAL FUND - UNKNOWN": "carrier_type_5d_special_fund_unknown",
        "Carrier Type_UNKNOWN": "carrier_type_unknown",
        
        # Geographic and Administrative
        "County of Injury": "county_of_injury",
        "District Name": "district_name",
        "Zip Code": "zip_code",
        
        # Injury and Industry Classification
        "WCIO Nature of Injury Code": "wcio_nature_of_injury_code",
        "Industry Code": "industry_code",
        "WCIO Cause of Injury Code": "wcio_cause_of_injury_code",
        "WCIO Part Of Body Code": "wcio_part_of_body_code",
        
        # Medical Fee Regions
        "Medical Fee Region_II": "medical_fee_region_ii",
        "Medical Fee Region_III": "medical_fee_region_iii",
        "Medical Fee Region_IV": "medical_fee_region_iv",
        "Medical Fee Region_UK": "medical_fee_region_uk",
        
        # Special Indicators
        "COVID-19 Indicator_Y": "covid_19_indicator_y",
        "Risk_Level": "risk_level",
        "Holiday_Accident": "holiday_accident",
        "Weekend_Accident": "weekend_accident",
        
        # Seasonal Features
        "Accident_Season_Sin": "accident_season_sin",
        "Accident_Season_Cos": "accident_season_cos",
        
        # Target (if present)
        "Claim Injury Type Encoded": "claim_injury_type_encoded"
    }
    
    # Create a copy of the data
    mapped_data = data.copy()
    
    # Rename columns that exist in the mapping
    existing_columns = {k: v for k, v in column_mapping.items() if k in mapped_data.columns}
    mapped_data = mapped_data.rename(columns=existing_columns)
    
    logger.info(f"Mapped {len(existing_columns)} column names for feature store compatibility")
    logger.debug(f"Mapped columns: {existing_columns}")
    
    # Log unmapped columns for debugging
    unmapped_columns = [col for col in mapped_data.columns if col not in existing_columns.values()]
    if unmapped_columns:
        logger.warning(f"Unmapped columns: {unmapped_columns}")
    
    # Convert data types for feature store compatibility
    mapped_data = convert_data_types_for_feature_store(mapped_data)
    
    return mapped_data

def convert_data_types_for_feature_store(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert data types to be compatible with feature store schemas.
    
    Args:
        data: DataFrame with mapped column names
        
    Returns:
        DataFrame with converted data types
    """
    # Create a copy to avoid modifying original
    converted_data = data.copy()
    
    # Define data type conversions for feature store compatibility
    type_conversions = {
        # Convert integer columns to float for feature store compatibility
        'age_at_injury': 'float64',
        'birth_year': 'float64',
        'average_weekly_wage': 'float64',
        'number_of_dependents': 'float64',
        'ime_4_count': 'float64',
        'days_to_first_hearing': 'float64',
        'days_to_c2': 'float64',
        'days_to_c3': 'float64',
        
        # Date components should be integers
        'c_2_date_year': 'int64',
        'c_2_date_month': 'int64',
        'c_2_date_day': 'int64',
        'c_2_date_dayofweek': 'int64',
        'c_3_date_year': 'int64',
        'c_3_date_month': 'int64',
        'c_3_date_day': 'int64',
        'c_3_date_dayofweek': 'int64',
        'first_hearing_date_year': 'int64',
        'first_hearing_date_month': 'int64',
        'first_hearing_date_day': 'int64',
        'first_hearing_date_dayofweek': 'int64',
        'accident_date_year': 'int64',
        'accident_date_month': 'int64',
        'accident_date_day': 'int64',
        'accident_date_dayofweek': 'int64',
        'assembly_date_year': 'int64',
        'assembly_date_month': 'int64',
        'assembly_date_day': 'int64',
        'assembly_date_dayofweek': 'int64',
        
        # Boolean indicators should be integers
        'known_age_at_injury': 'int64',
        'known_birth_year': 'int64',
        'known_c_2_date': 'int64',
        'known_c_3_date': 'int64',
        'known_first_hearing_date': 'int64',
        'known_accident_date': 'int64',
        'known_assembly_date': 'int64',
        'attorney_representative_y': 'int64',
        'alternative_dispute_resolution_y': 'int64',
        'alternative_dispute_resolution_u': 'int64',
        'carrier_type_2a_sif': 'int64',
        'carrier_type_3a_self_public': 'int64',
        'carrier_type_4a_self_private': 'int64',
        'carrier_type_5d_special_fund_unknown': 'int64',
        'carrier_type_unknown': 'int64',
        'covid_19_indicator_y': 'int64',
        'holiday_accident': 'int64',
        'weekend_accident': 'int64',
        
        # String columns
        'county_of_injury': 'string',
        'district_name': 'string',
        'zip_code': 'string',
        'wcio_nature_of_injury_code': 'string',
        'industry_code': 'string',
        'wcio_cause_of_injury_code': 'string',
        'wcio_part_of_body_code': 'string',
        'medical_fee_region_ii': 'string',
        'medical_fee_region_iii': 'string',
        'medical_fee_region_iv': 'string',
        'medical_fee_region_uk': 'string',
        'risk_level': 'string',
        
        # Float columns
        'relative_wage': 'float64',
        'accident_season_sin': 'float64',
        'accident_season_cos': 'float64',
    }
    
    # Apply type conversions for columns that exist
    for col, target_type in type_conversions.items():
        if col in converted_data.columns:
            try:
                if target_type == 'string':
                    converted_data[col] = converted_data[col].astype('string')
                elif target_type == 'int64':
                    # Handle NaN values for integer conversion
                    converted_data[col] = pd.to_numeric(converted_data[col], errors='coerce').fillna(0).astype('int64')
                elif target_type == 'float64':
                    converted_data[col] = pd.to_numeric(converted_data[col], errors='coerce').astype('float64')
                logger.debug(f"Converted {col} to {target_type}")
            except Exception as e:
                logger.warning(f"Failed to convert {col} to {target_type}: {e}")
    
    logger.info("Data type conversion completed for feature store compatibility")
    return converted_data

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
    logger.info(f"Original data shape: {data.shape}")
    
    # Map column names to snake_case for feature store compatibility
    data = map_column_names(data)
    logger.info(f"Data shape after column mapping: {data.shape}")
    
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

def validate_feature_store_setup(
    data: pd.DataFrame,
    feature_store_params: Dict[str, Any],
    feature_groups_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate feature store setup and create expectation suites for all feature groups.
    
    Args:
        data: Input dataframe with all features
        feature_store_params: Feature store connection parameters
        feature_groups_config: Configuration for feature groups
        
    Returns:
        Dict with validation results and setup information
    """
    
    logger.info("=== Feature Store Setup Validation ===")
    
    # Map column names
    mapped_data = map_column_names(data)
    
    validation_results = {
        "original_columns": list(data.columns),
        "mapped_columns": list(mapped_data.columns),
        "feature_groups_analysis": {},
        "gx_setup": {},
        "recommendations": []
    }
    
    # Analyze each feature group
    for group_name, config in feature_groups_config.items():
        available_columns = [col for col in config["columns"] if col in mapped_data.columns]
        missing_columns = [col for col in config["columns"] if col not in mapped_data.columns]
        
        validation_results["feature_groups_analysis"][group_name] = {
            "available_columns": available_columns,
            "missing_columns": missing_columns,
            "coverage_percentage": len(available_columns) / len(config["columns"]) * 100 if config["columns"] else 0,
            "can_create": len(available_columns) > 0
        }
        
        if missing_columns:
            validation_results["recommendations"].append(
                f"Feature group '{group_name}' is missing columns: {missing_columns}"
            )
    
    # Test Great Expectations setup
    try:
        gx_context = get_gx_context()
        validation_results["gx_setup"]["status"] = "connected"
        validation_results["gx_setup"]["context_type"] = type(gx_context).__name__
        
        # Create expectation suites for each feature group
        for group_name, config in feature_groups_config.items():
            if validation_results["feature_groups_analysis"][group_name]["can_create"]:
                suite_name = f"{group_name}_expectations"
                try:
                    suite = gx_context.create_expectation_suite(suite_name)
                    
                    # Add basic expectations based on available columns
                    available_columns = validation_results["feature_groups_analysis"][group_name]["available_columns"]
                    
                    for col in available_columns:
                        # Add column existence expectation
                        suite.add_expectation({
                            "expectation_type": "expect_column_to_exist",
                            "kwargs": {"column": col}
                        })
                        
                        # Add data type expectations based on column name patterns
                        if any(pattern in col.lower() for pattern in ["count", "number", "days", "age", "wage"]):
                            suite.add_expectation({
                                "expectation_type": "expect_column_values_to_be_between",
                                "kwargs": {
                                    "column": col,
                                    "min_value": 0,
                                    "max_value": 999999
                                }
                            })
                        elif any(pattern in col.lower() for pattern in ["gender", "known", "covid", "attorney"]):
                            suite.add_expectation({
                                "expectation_type": "expect_column_values_to_be_in_set",
                                "kwargs": {
                                    "column": col,
                                    "value_set": [0, 1]
                                }
                            })
                    
                    gx_context.save_expectation_suite(suite)
                    logger.info(f"✅ Created expectation suite: {suite_name}")
                    
                except Exception as e:
                    logger.warning(f"⚠️  Could not create expectation suite for {group_name}: {e}")
                    validation_results["gx_setup"]["errors"] = validation_results["gx_setup"].get("errors", [])
                    validation_results["gx_setup"]["errors"].append(f"{group_name}: {str(e)}")
        
    except Exception as e:
        validation_results["gx_setup"]["status"] = "failed"
        validation_results["gx_setup"]["error"] = str(e)
        validation_results["recommendations"].append(f"Great Expectations setup failed: {e}")
    
    # Test feature store connection
    try:
        feature_store = connect_to_feature_store(
            api_key=feature_store_params["api_key"],
            project_name=feature_store_params["project_name"]
        )
        validation_results["feature_store_connection"] = "success"
        logger.info("✅ Feature store connection successful")
    except Exception as e:
        validation_results["feature_store_connection"] = "failed"
        validation_results["feature_store_error"] = str(e)
        validation_results["recommendations"].append(f"Feature store connection failed: {e}")
    
    # Generate summary
    total_groups = len(feature_groups_config)
    creatable_groups = sum(1 for analysis in validation_results["feature_groups_analysis"].values() 
                          if analysis["can_create"])
    
    validation_results["summary"] = {
        "total_feature_groups": total_groups,
        "creatable_feature_groups": creatable_groups,
        "gx_available": validation_results["gx_setup"]["status"] == "connected",
        "feature_store_available": validation_results["feature_store_connection"] == "success",
        "overall_status": "ready" if (creatable_groups > 0 and 
                                    validation_results["feature_store_connection"] == "success") else "issues"
    }
    
    logger.info(f"=== VALIDATION SUMMARY ===")
    logger.info(f"📊 Total feature groups: {total_groups}")
    logger.info(f"✅ Creatable feature groups: {creatable_groups}")
    logger.info(f"🔍 Great Expectations: {validation_results['gx_setup']['status']}")
    logger.info(f"🏪 Feature Store: {validation_results['feature_store_connection']}")
    logger.info(f"📋 Overall Status: {validation_results['summary']['overall_status']}")
    
    if validation_results["recommendations"]:
        logger.info("📝 Recommendations:")
        for rec in validation_results["recommendations"]:
            logger.info(f"   - {rec}")
    
    return validation_results