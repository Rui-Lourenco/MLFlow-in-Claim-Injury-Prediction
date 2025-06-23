from kedro.pipeline import Pipeline, pipeline, node
from .nodes import train_and_evaluate_models, save_trained_models, generate_predictions

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_and_evaluate_models,
                inputs=dict(
                    X_train="X_train_selected",
                    X_val="X_val_selected",
                    X_test="X_test_selected",
                    y_train="y_train",
                    y_val="y_val",
                    y_test="y_test",
                    xgb_params="params:xgb_params",
                    rf_params="params:rf_params"
                ),
                outputs=["models_dict", "best_model_name", "best_model", "test_metrics", "test_report", "feature_importance"],
                name="train_and_evaluate_models_node"
            ),
            node(
                func=save_trained_models,
                inputs=dict(
                    models_dict="models_dict",
                    best_model_name="best_model_name",
                    model_save_path="params:model_save_path"
                ),
                outputs="best_model_path",
                name="save_trained_models_node"
            ),
            node(
                func=generate_predictions,
                inputs=dict(
                    best_model="best_model",
                    X_test="X_test_selected",
                    y_test="y_test"
                ),
                outputs=["predictions", "probabilities", "results_df", "confusion_matrix"],
                name="generate_predictions_node"
            ),
        ]
    ) 