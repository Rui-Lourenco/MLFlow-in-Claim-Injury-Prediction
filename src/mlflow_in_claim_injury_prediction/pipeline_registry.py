"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline

# Import pipelines directly
from .pipelines import data_units_test, feature_store

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.
    
    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    
    # Manual registration (bypassing find_pipelines() which has issues)
    pipelines = {}
    
    # Add individual pipelines
    pipelines["data_units_tests"] = data_units_test.create_pipeline()
    pipelines["feature_store"] = feature_store.create_pipeline()
    
    
    
    return pipelines
