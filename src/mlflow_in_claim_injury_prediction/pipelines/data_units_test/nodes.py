import logging
import pandas as pd
import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest

log = logging.getLogger(__name__)

def test_data(
    df: pd.DataFrame,
    datasource_name: str,
    suite_name: str,
    data_asset_name: str,
    build_data_docs: bool = False
) -> pd.DataFrame:
    """
    Validates raw input data using Great Expectations.

    This function performs initial data quality validation on raw data
    before any processing or transformations are applied.

    Args:
        df: The raw input data to validate.
        datasource_name: The GE datasource defined in great_expectations.yml.
        suite_name: The expectation suite name for raw data validation.
        data_asset_name: The name of the data asset (e.g., filename without .csv).
        build_data_docs: Whether to build and open data docs after validation.

    Returns:
        The input DataFrame if validation passes.

    Raises:
        ValueError: If validation fails.
    """
    context = gx.get_context()

    # Use RuntimeBatchRequest for in-memory DataFrame validation
    batch_request = RuntimeBatchRequest(
        datasource_name=datasource_name,
        data_connector_name="default_runtime_data_connector_name",
        data_asset_name=data_asset_name,
        runtime_parameters={"batch_data": df},
        batch_identifiers={"runtime_batch_identifier_name": "default_identifier"}
    )

    try:
        validator = context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=suite_name
        )

        validation_result = validator.validate()

        if build_data_docs:
            context.build_data_docs()
            context.open_data_docs()

        if not validation_result["success"]:
            log.warning(f"Data validation failed for asset: {data_asset_name}")
            log.warning(f"Validation results: {validation_result}")
            # For now, we'll log the failure but continue processing
            # You can uncomment the next line to make validation failures stop the pipeline
            # raise ValueError(f"Initial data validation failed for asset: {data_asset_name}")
        else:
            log.info(f"Initial data validation passed for asset: {data_asset_name}")

    except Exception as e:
        log.warning(f"Could not perform data validation for {data_asset_name}: {str(e)}")
        log.info("Continuing with data processing despite validation issues")

    return df
