"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline

# Import pipelines directly
from .pipelines import data_units_test, feature_store, data_upload, data_drift, data_preparation, data_units_tests_after_processing, data_split, data_processing, feature_engineering, feature_selection, model_inference

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.
    
    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    
    pipelines = {}
    
    # Add individual pipelines with clear naming
    pipelines["initial_data_validation"] = data_units_test.create_pipeline()  # Raw data validation
    pipelines["final_data_validation"] = data_units_tests_after_processing.create_pipeline()  # Final processed data validation
    pipelines["feature_store"] = feature_store.create_pipeline()
    pipelines["data_upload"] = data_upload.create_pipeline()
    pipelines["data_drift"] = data_drift.create_pipeline()
    pipelines["data_preparation"] = data_preparation.create_pipeline()
    pipelines["data_split"] = data_split.create_pipeline()
    pipelines["data_processing"] = data_processing.create_pipeline()
    pipelines["feature_engineering"] = feature_engineering.create_pipeline()
    pipelines["feature_selection"] = feature_selection.create_pipeline()
    pipelines["model_inference"] = model_inference.create_pipeline()
    
    # Create a complete pipeline that combines all steps in logical order
    complete_pipeline = (
        data_units_test.create_pipeline() +                    # 1. Initial raw data validation
        data_preparation.create_pipeline() +                   # 2. Data preparation
        data_split.create_pipeline() +                         # 3. Data splitting
        data_processing.create_pipeline() +                    # 4. Data processing (scaling, imputation)
        feature_engineering.create_pipeline() +                # 5. Feature engineering
        feature_selection.create_pipeline() +                  # 6. Feature selection
        data_units_tests_after_processing.create_pipeline() +  # 7. Final data validation (after feature selection)
        feature_store.create_pipeline() +                      # 8. Feature store operations
        data_upload.create_pipeline() +                        # 9. Data upload
        data_drift.create_pipeline() +                         # 10. Data drift detection
        model_inference.create_pipeline()                      # 11. Model training and inference
    )
    
    pipelines["complete_pipeline"] = complete_pipeline
    
    # Keep legacy names for backward compatibility
    pipelines["data_units_tests"] = data_units_test.create_pipeline()
    pipelines["data_units_tests_after_processing"] = data_units_tests_after_processing.create_pipeline()
    
    return pipelines
