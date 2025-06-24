from kedro.pipeline import Pipeline, pipeline, node
from .nodes import encode_data

def create_pipeline(**kwargs) -> Pipeline:
    """Create the data preparation pipeline."""
    return pipeline(
        [
            node(
                func=encode_data,
                inputs=dict(
                    data="processed_data_validated_final",
                    target_column="params:target_column",
                    date_columns="params:date_columns"
                ),
                outputs="data_encoded",
                name="encode_data_node"
            ),
        ]
    ) 