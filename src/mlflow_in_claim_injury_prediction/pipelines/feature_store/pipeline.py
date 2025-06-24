



from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_feature_groups_with_gx_robust

def create_pipeline(**kwargs) -> Pipeline:
    """Create the robust feature store pipeline with Great Expectations."""
    return pipeline([
        node(
            func=create_feature_groups_with_gx_robust,
            inputs=[
                "model_input_data",
                "params:feature_store", 
                "params:feature_groups_config"
            ],
            outputs="feature_groups_metadata",
            name="robust_feature_groups_with_gx",
        ),
    ])

# ============================================================================
# Example: Creating Expectation Suites for Your Feature Groups
# ============================================================================

def setup_feature_group_expectations():
    """
    Example function to create expectation suites for your feature groups.
    Run this once to set up expectations in your existing GX setup.
    """
    
    context = gx.get_context()
    
    # Example: Create expectations for personal information feature group
    suite_name = "personal_information_expectations"
    suite = context.create_expectation_suite(suite_name)
    
    # Add some basic expectations
    suite.add_expectation({
        "expectation_type": "expect_column_to_exist",
        "kwargs": {"column": "Age at Injury"}
    })
    
    suite.add_expectation({
        "expectation_type": "expect_column_values_to_be_between",
        "kwargs": {
            "column": "Age at Injury",
            "min_value": 16,
            "max_value": 100
        }
    })
    
    suite.add_expectation({
        "expectation_type": "expect_column_values_to_be_in_set",
        "kwargs": {
            "column": "Gender_M",
            "value_set": [0, 1]
        }
    })
    
    # Save the suite
    context.save_expectation_suite(suite)
    
    # Repeat for other feature groups...
    print(f"Created expectation suite: {suite_name}")
