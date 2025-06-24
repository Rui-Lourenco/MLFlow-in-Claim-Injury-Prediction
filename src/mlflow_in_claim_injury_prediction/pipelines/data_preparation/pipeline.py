from kedro.pipeline import Pipeline, pipeline, node
from .nodes import prepare_data_for_splitting

def create_pipeline(**kwargs) -> Pipeline:
    """Create the data preparation pipeline."""
    return pipeline([
        node(
            func=prepare_data_for_splitting,
            inputs=dict(
                data="processed_data",
                target_column="params:target_column"
            ),
            outputs="processed_data_validated_final",
            name="prepare_data_for_splitting",
        ),
    ]) 