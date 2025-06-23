"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline

# Import pipelines directly
from .pipelines import data_units_test, feature_store, data_units_tests_after_processing, data_split, data_processing, feature_engineering, feature_selection, model_inference

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.
    
    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    
    pipelines = {}
    
    # Add individual pipelines
    pipelines["data_units_tests"] = data_units_test.create_pipeline()
    pipelines["feature_store"] = feature_store.create_pipeline()
    pipelines["data_units_tests_after_processing"] = data_units_tests_after_processing.create_pipeline()
    pipelines["data_split"] = data_split.create_pipeline()
    pipelines["data_processing"] = data_processing.create_pipeline()
    pipelines["feature_engineering"] = feature_engineering.create_pipeline()
    pipelines["feature_selection"] = feature_selection.create_pipeline()
    pipelines["model_inference"] = model_inference.create_pipeline()
    
    # Create a complete pipeline that combines all steps
    complete_pipeline = (
        data_units_test.create_pipeline() +
        data_units_tests_after_processing.create_pipeline() +
        data_split.create_pipeline() +
        data_processing.create_pipeline() +
        feature_engineering.create_pipeline() +
        feature_selection.create_pipeline() +
        model_inference.create_pipeline()
    )
    
    pipelines["complete_pipeline"] = complete_pipeline
    
    return pipelines
