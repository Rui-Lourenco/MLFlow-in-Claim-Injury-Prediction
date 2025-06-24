from kedro.pipeline import Pipeline, pipeline, node
from .nodes import calculate_permutation_importance


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=calculate_permutation_importance,
            inputs=[
                "best_model", 
                "X_train_selected", 
                "y_train",
                "params:explainability.n_repeats",
                "params:explainability.random_state"
            ],
            outputs="permutation_importance"
        )
    ]) 