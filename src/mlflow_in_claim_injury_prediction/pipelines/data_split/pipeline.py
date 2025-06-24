from kedro.pipeline import Pipeline, pipeline, node
from .nodes import split_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=dict(
                    df="processed_data_validated_final",
                    target_column="params:target_column",
                    test_size="params:test_size",
                    val_size="params:val_size",
                    random_state="params:random_state"
                ),
                outputs=["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"],
                name="split_data_node"
            ),
        ]
    ) 