from kedro.pipeline import Pipeline, pipeline, node
from .nodes import preprocess_raw_data

def create_pipeline(**kwargs) -> Pipeline:
    """Create the data preprocessing pipeline."""
    return pipeline([
        node(
            func=preprocess_raw_data,
            inputs="raw_data_validated",
            outputs="processed_data",
            name="preprocess_raw_data_node",
        ),
    ]) 