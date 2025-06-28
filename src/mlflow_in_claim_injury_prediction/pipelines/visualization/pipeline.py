"""Visualization pipeline for MLOps project."""

from kedro.pipeline import Pipeline, node
from .nodes import (
    create_feature_importance_comparison,
    create_confusion_matrix_heatmap,
    create_model_performance_dashboard,
    create_data_drift_visualization,
    create_permutation_importance_plot
)

def create_pipeline(**kwargs) -> Pipeline:
    """Create the visualization pipeline."""
    
    return Pipeline([
        node(
            func=create_feature_importance_comparison,
            inputs="feature_importance_summary",
            outputs=None,  # Files created directly, no catalog output
            name="create_feature_importance_comparison_node"
        ),
        node(
            func=create_confusion_matrix_heatmap,
            inputs="confusion_matrix",
            outputs=None,  # Files created directly, no catalog output
            name="create_confusion_matrix_heatmap_node"
        ),
        node(
            func=create_model_performance_dashboard,
            inputs=["test_metrics", "test_report"],
            outputs=None,  # Files created directly, no catalog output
            name="create_performance_dashboard_node"
        ),
        node(
            func=create_data_drift_visualization,
            inputs="data_drift_report",
            outputs=None,  # Files created directly, no catalog output
            name="create_drift_visualization_node"
        ),
        node(
            func=create_permutation_importance_plot,
            inputs="permutation_importance",
            outputs=None,  # Files created directly, no catalog output
            name="create_permutation_importance_plot_node"
        )
    ])