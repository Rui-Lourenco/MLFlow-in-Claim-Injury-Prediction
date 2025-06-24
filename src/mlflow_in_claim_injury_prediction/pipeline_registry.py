"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline

# Import pipelines directly
from .pipelines import data_units_test, feature_store, data_upload, data_drift, data_preparation, data_units_tests_after_processing, data_split, data_transformations, feature_engineering, feature_selection, model_inference, data_preprocessing

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.
    
    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    
    pipelines = {}
    
    # Add individual pipelines with clear naming
    pipelines["initial_data_validation"] = data_units_test.create_pipeline()  # Raw data validation
    pipelines["data_preprocessing"] = data_preprocessing.create_pipeline()  # Raw to processed data conversion
    pipelines["final_data_validation"] = data_units_tests_after_processing.create_pipeline()  # Final processed data validation
    pipelines["feature_store"] = feature_store.create_pipeline()
    pipelines["data_upload"] = data_upload.create_pipeline()
    pipelines["data_drift"] = data_drift.create_pipeline()
    pipelines["data_preparation"] = data_preparation.create_pipeline()
    pipelines["data_transformations"] = data_transformations.create_pipeline()
    pipelines["data_split"] = data_split.create_pipeline()
    pipelines["feature_engineering"] = feature_engineering.create_pipeline()
    pipelines["feature_selection"] = feature_selection.create_pipeline()
    pipelines["model_inference"] = model_inference.create_pipeline()
    
    # Create a simplified pipeline with correct order based on notebooks
    simplified_pipeline = (
        data_units_test.create_pipeline() +                    # 1. Initial raw data validation (EDA notebook)
        data_preprocessing.create_pipeline() +                 # 2. Basic preprocessing (data types, cleaning)
        data_preparation.create_pipeline() +                   # 3. Data encoding (target, one-hot, sine-cosine, date components)
        feature_engineering.create_pipeline() +                # 4. Advanced feature engineering (interactions, polynomials, statistical)
        data_split.create_pipeline() +                         # 5. Data splitting (after all pre-split processing)
        data_transformations.create_pipeline() +               # 6. Post-split processing (frequency encoding, imputation, feature creation)
        feature_selection.create_pipeline() +                  # 7. Feature selection
        data_units_tests_after_processing.create_pipeline() +  # 8. Final data validation
        model_inference.create_pipeline()                      # 9. Model training and inference
    )
    
    # Create a complete pipeline that combines all steps in logical order
    complete_pipeline = (
        data_units_test.create_pipeline() +                    # 1. Initial raw data validation (EDA notebook)
        data_preprocessing.create_pipeline() +                 # 2. Basic preprocessing (data types, cleaning)
        data_preparation.create_pipeline() +                   # 3. Data preparation (target encoding, duplicates removal)
        feature_engineering.create_pipeline() +                # 4. Feature engineering BEFORE splitting (date components, seasonality, etc.)
        data_split.create_pipeline() +                         # 5. Data splitting (after feature engineering)
        data_transformations.create_pipeline() +               # 6. Post-split processing (frequency encoding, imputation, feature creation)
        feature_selection.create_pipeline() +                  # 7. Feature selection
        data_units_tests_after_processing.create_pipeline() +  # 8. Final data validation
        feature_store.create_pipeline() +                      # 9. Feature store operations
        data_upload.create_pipeline() +                        # 10. Data upload
        data_drift.create_pipeline() +                         # 11. Data drift detection
        model_inference.create_pipeline()                      # 12. Model training and inference
    )
    
    pipelines["simplified_pipeline"] = simplified_pipeline
    pipelines["complete_pipeline"] = complete_pipeline
    
    # Keep legacy names for backward compatibility
    pipelines["data_units_tests"] = data_units_test.create_pipeline()
    pipelines["data_units_tests_after_processing"] = data_units_tests_after_processing.create_pipeline()
    pipelines["data_processing"] = data_transformations.create_pipeline()  # Legacy name for backward compatibility
    
    return pipelines
