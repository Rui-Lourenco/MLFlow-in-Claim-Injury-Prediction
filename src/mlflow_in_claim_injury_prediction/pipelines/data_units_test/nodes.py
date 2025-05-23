import logging
import pandas as pd
import great_expectations as gx
from great_expectations.core.batch import BatchRequest

log = logging.getLogger(__name__)

def test_data(
    df: pd.DataFrame,
    datasource_name: str,
    suite_name: str,
    data_asset_name: str,
    build_data_docs: bool = False
) -> pd.DataFrame:
    """
    Validates a pandas DataFrame using Great Expectations.

    Args:
        df: The data to validate.
        datasource_name: The GE datasource defined in great_expectations.yml.
        suite_name: The expectation suite name.
        data_asset_name: The name of the data asset (e.g., filename without .csv).
        build_data_docs: Whether to build and open data docs after validation.

    Returns:
        The input DataFrame if validation passes.

    Raises:
        ValueError: If validation fails.
    """
    context = gx.get_context()

    batch_request = {
        'datasource_name': datasource_name,
        'data_connector_name': "default_inferred_data_connector_name",
        'data_asset_name': f"{data_asset_name}.csv",
        'batch_spec_passthrough': {"reader_method": "read_csv"}
    }

    validator = context.get_validator(
        batch_request=BatchRequest(**batch_request),
        expectation_suite_name=suite_name
    )

    validation_result = validator.validate()

    if build_data_docs:
        context.build_data_docs()
        context.open_data_docs()

    if not validation_result["success"]:
        raise ValueError(f"Data validation failed for asset: {data_asset_name}.csv")

    log.info(f"Data validation passed for asset: {data_asset_name}.csv")
    return df
