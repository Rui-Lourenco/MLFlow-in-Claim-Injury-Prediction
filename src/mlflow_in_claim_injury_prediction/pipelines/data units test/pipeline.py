from kedro.pipeline import Pipeline, pipeline, node
from .nodes import test_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=test_data,
                inputs=dict(
                    df="raw_data_pre_validation",
                    datasource_name="params:raw_datasource_name",
                    suite_name="params:raw_suite_name",
                    data_asset_name="params:raw_data_asset_name",
                    build_data_docs="params:build_data_docs"
                ),
                outputs="raw_data_validated",
                name="validate_raw_data_node"
            ),
            node(
                func=test_data,
                inputs=dict(
                    df="processed_data_pre_validation",
                    datasource_name="params:processed_datasource_name",
                    suite_name="params:processed_suite_name",
                    data_asset_name="params:processed_data_asset_name",
                    build_data_docs="params:build_data_docs"
                ),
                outputs="processed_data_validated",
                name="validate_processed_data_node"
            ),
        ]
    )
