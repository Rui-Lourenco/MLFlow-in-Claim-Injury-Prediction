from kedro.pipeline import Pipeline, pipeline, node
from .nodes import test_data

def create_pipeline(**kwargs) -> Pipeline:
    """Create the initial data validation pipeline for raw data only."""
    return pipeline(
        [
            node(
                func=test_data,
                inputs=dict(
                    df="raw_input_data",
                    datasource_name="params:raw_datasource_name",
                    suite_name="params:raw_suite_name",
                    data_asset_name="params:raw_data_asset_name",
                    build_data_docs="params:build_data_docs"
                ),
                outputs="raw_data_validated",
                name="validate_raw_data_node"
            ),
        ],
        inputs=["raw_input_data"],
        outputs=["raw_data_validated"]
    )