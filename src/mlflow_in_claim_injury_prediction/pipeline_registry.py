"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline

# Import pipelines directly
from .pipelines import (
    data_units_test,
    data_preprocessing, 
    data_preparation,
    data_split,
    data_transformations,
    feature_engineering,
    feature_selection,
    model_inference,
    data_units_tests_after_processing,
    explainability,
    data_drift
)

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.
    
    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    
    pipelines = {
        "data_validation": data_units_test.create_pipeline(),
        "data_preprocessing": data_preprocessing.create_pipeline(),
        "data_preparation": data_preparation.create_pipeline(),
        "data_split": data_split.create_pipeline(),
        "data_transformations": data_transformations.create_pipeline(),
        "feature_engineering": feature_engineering.create_pipeline(),
        "feature_selection": feature_selection.create_pipeline(),
        "final_validation": data_units_tests_after_processing.create_pipeline(),
        "model_inference": model_inference.create_pipeline(),
        "explainability": explainability.create_pipeline(),
        "data_drift": data_drift.create_pipeline(),
    }
    
    pipelines["training_pipeline"] = (
        data_units_test.create_pipeline() +
        data_preprocessing.create_pipeline() +
        data_preparation.create_pipeline() +
        feature_engineering.create_pipeline() +
        data_units_tests_after_processing.create_pipeline() +
        data_split.create_pipeline() +
        data_transformations.create_pipeline() +
        feature_selection.create_pipeline() +
        model_inference.create_pipeline() +
        explainability.create_pipeline() +
        data_drift.create_pipeline()
    )
    
    pipelines["inference_pipeline"] = (
        data_preprocessing.create_pipeline() +
        data_preparation.create_pipeline() +
        feature_engineering.create_pipeline() +
        data_transformations.create_pipeline() +
        model_inference.create_pipeline()
    )
    
    pipelines["__default__"] = pipelines["training_pipeline"]
    
    return pipelines
