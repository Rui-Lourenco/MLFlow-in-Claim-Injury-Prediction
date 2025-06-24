from kedro.pipeline import Pipeline, pipeline, node
from .nodes import test_processed_data

def create_pipeline(**kwargs) -> Pipeline:
    """Create the final data validation pipeline for processed data after all transformations."""
    return pipeline(
        [
            node(
                func=test_processed_data,
                inputs=dict(
                    df="data_engineered",  # Validate the engineered data after feature engineering
                    datasource_name="params:processed_datasource_name",
                    suite_name="params:processed_suite_name",
                    data_asset_name="params:processed_data_asset_name",
                    build_data_docs="params:build_data_docs"
                ),
                outputs="final_processed_data_validated",
                name="validate_final_processed_data_node"
            ),
        ],
        inputs=["data_engineered"],
        outputs=["final_processed_data_validated"]
    ) 