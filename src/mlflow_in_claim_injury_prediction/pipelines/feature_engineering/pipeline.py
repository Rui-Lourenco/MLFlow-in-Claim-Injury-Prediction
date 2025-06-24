from kedro.pipeline import Pipeline, pipeline, node
from .nodes import engineer_features

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=engineer_features,
                inputs=dict(
                    data="data_encoded",
                    create_polynomial_features="params:create_polynomial_features",
                    polynomial_degree="params:polynomial_degree"
                ),
                outputs="data_engineered",
                name="engineer_features_node"
            ),
        ]
    ) 