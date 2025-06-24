from kedro.pipeline import Pipeline, pipeline, node
from .nodes import detect_data_drift


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data drift detection pipeline."""
    return pipeline([
        node(
            func=detect_data_drift,
            inputs=[
                "X_train_selected",  # Reference data (training data)
                "X_test_selected",   # Current data to check for drift (test data)
                "params:numerical_features",
                "params:categorical_features"
            ],
            outputs="data_drift_report",
            name="detect_data_drift",
        ),
    ]) 