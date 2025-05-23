"""Project pipelines."""

from kedro.pipeline import Pipeline
from .pipelines import (
    data_units_test,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_units_test_pipeline= data_units_test.create_pipeline()
    return  {"data_units_tests": data_units_test_pipeline}
