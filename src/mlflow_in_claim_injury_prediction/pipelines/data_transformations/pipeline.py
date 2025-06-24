from kedro.pipeline import Pipeline, pipeline, node
from .nodes import apply_processing

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=apply_processing,
            inputs=dict(
                X_train="X_train",
                X_val="X_val", 
                X_test="X_test"
            ),
            outputs=["X_train_final", "X_val_final", "X_test_final"],
            name="apply_post_split_processing_node"
        ),
    ]) 