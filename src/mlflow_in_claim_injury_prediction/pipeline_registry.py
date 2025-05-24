"""Project pipelines."""

from typing import Dict
from kedro.pipeline import Pipeline
from kedro.framework.project import find_pipelines

# Import pipelines
from .pipelines import data_units_test, feature_store

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()

    # Add individual pipelines
    pipelines["data_units_tests"] = data_units_test.create_pipeline()
    pipelines["feature_store"] = feature_store.create_pipeline()

    return pipelines
