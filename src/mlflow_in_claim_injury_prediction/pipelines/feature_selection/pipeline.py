from kedro.pipeline import Pipeline, pipeline, node
from .nodes import combine_feature_selection

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=combine_feature_selection,
                inputs=dict(
                    X_train="X_train_final",
                    X_val="X_val_final",
                    X_test="X_test_final",
                    y_train="y_train",
                    y_val="y_val",
                    xgb_threshold="params:xgb_threshold",
                    rf_threshold="params:rf_threshold",
                    max_features="params:max_features",
                    selection_method="params:selection_method"
                ),
                outputs=["X_train_selected", "X_val_selected", "X_test_selected", "selected_features", "feature_importance_summary"],
                name="combine_feature_selection_node"
            ),
        ]
    ) 