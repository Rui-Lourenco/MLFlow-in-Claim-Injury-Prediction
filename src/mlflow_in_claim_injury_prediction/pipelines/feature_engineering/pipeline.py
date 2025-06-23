from kedro.pipeline import Pipeline, pipeline, node
from .nodes import engineer_features

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=engineer_features,
                inputs=dict(
                    X_train="X_train_final",
                    X_val="X_val_final",
                    X_test="X_test_final",
                    date_columns="params:date_columns",
                    create_polynomial_features="params:create_polynomial_features",
                    polynomial_degree="params:polynomial_degree"
                ),
                outputs=["X_train_engineered", "X_val_engineered", "X_test_engineered", "feature_names"],
                name="engineer_features_node"
            ),
        ]
    ) 