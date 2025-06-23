from kedro.pipeline import Pipeline, pipeline, node
from .nodes import upload_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=upload_data,
            inputs=[
                "processed_data",
                "params:group_name",
                "params:description",
                "params:feature_descriptions",
                "params:suite_name"
            ],
            outputs=None,
            name="upload_data",
        )
    ]) 