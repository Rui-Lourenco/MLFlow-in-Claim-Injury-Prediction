from kedro.pipeline import Pipeline, pipeline, node
from .nodes import process_data, apply_advanced_processing

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=process_data,
                inputs=dict(
                    X_train="X_train",
                    X_val="X_val",
                    X_test="X_test",
                    numerical_features="params:numerical_features",
                    categorical_features="params:categorical_features",
                    scaling_method="params:scaling_method",
                    imputation_method="params:imputation_method"
                ),
                outputs=["X_train_processed", "X_val_processed", "X_test_processed", "scaler", "imputer"],
                name="process_data_node"
            ),
            node(
                func=apply_advanced_processing,
                inputs=dict(
                    X_train="X_train_processed",
                    X_val="X_val_processed",
                    X_test="X_test_processed"
                ),
                outputs=["X_train_final", "X_val_final", "X_test_final"],
                name="apply_advanced_processing_node"
            ),
        ]
    ) 