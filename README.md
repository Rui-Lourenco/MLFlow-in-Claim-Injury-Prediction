# MLFlow in Claim Injury Prediction

## Overview

This project implements a comprehensive machine learning pipeline for predicting injury outcomes from insurance claims data using Kedro and MLflow. The system includes data validation, feature engineering, model training, interpretability analysis, data drift detection, and deployment capabilities with automated MLflow tracking.

## Project Structure

```
MLFlow-in-Claim-Injury-Prediction/
├── conf/                          # Configuration files
│   ├── base/                      # Base configuration
│   │   ├── catalog.yml            # Data catalog definitions
│   │   ├── parameters.yml         # Pipeline parameters
│   │   └── mlflow.yml            # MLflow configuration
│   └── local/                     # Local configuration (not in git)
├── data/                          # Data files
│   ├── 01_raw/                    # Raw data
│   ├── 02_intermediate/           # Intermediate data
│   ├── 03_primary/                # Primary data
│   ├── 04_model_input/            # Model input data
│   ├── 05_context/                # Context for the notebooks
│   ├── 06_models/                 # Trained models
│   ├── 07_model_output/           # Model results
│   └── 08_reporting/              # Reports and outputs
├── docs/                          # Documentation
├── gx/                           # Great Expectations configuration
│   ├── expectations/              # Data validation suites
│   └── great_expectations.yml     # GE configuration
├── notebooks/                     # Jupyter notebooks for EDA and analysis
├── src/                           # Source code
│   └── mlflow_in_claim_injury_prediction/
│       ├── pipelines/             # Kedro pipelines
│       │   ├── data_units_test/   # Raw data validation
│       │   ├── data_preprocessing/ # Data preprocessing
│       │   ├── data_preparation/  # Data preparation
│       │   ├── data_split/        # Train/val/test split
│       │   ├── data_transformations/ # Feature scaling and imputation
│       │   ├── feature_engineering/ # Feature creation
│       │   ├── feature_selection/ # Feature selection
│       │   ├── model_inference/   # Model training and evaluation
│       │   ├── explainability/    # Model interpretability
│       │   ├── data_drift/        # Data drift detection
│       │   └── data_units_tests_after_processing/ # Processed data validation
│       └── utils/                 # Utility functions
│           ├── mlflow_utils.py    # MLflow integration utilities
│           └── feature_store_utils.py # Feature store utilities
├── tests/                         # Test files
├── utils/                         # Core utility modules
│   ├── model.py                   # Model training and evaluation
│   ├── utils.py                   # Data processing utilities
│   ├── utils_feature_selection.py # Feature selection utilities
│   └── utils_dicts.py             # Feature definitions
└── requirements.txt               # Python dependencies
```

## Key Features

- **Data Validation**: Great Expectations integration for comprehensive data quality checks
- **Feature Engineering**: Advanced feature creation including date components, interactions, and statistical features
- **Model Training**: XGBoost and Random Forest with automated MLflow tracking
- **Model Deployment**: MLflow model registry integration for versioning and deployment
- **Model Interpretability**: Permutation importance analysis for feature understanding
- **Data Drift Detection**: Statistical analysis to detect distribution changes between training and test data
- **Monitoring**: Comprehensive model performance tracking and drift monitoring
- **Automated Pipeline**: Complete end-to-end pipeline from raw data to model deployment

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd MLFlow-in-Claim-Injury-Prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup local configuration**:
   ```bash
   # Create local configuration directory
   mkdir -p conf/local
   
   # Copy and modify configuration files as needed
   cp conf/base/mlflow.yml conf/local/
   ```

## Usage

### Running the Pipeline

**Complete training pipeline**:
```bash
kedro run
```

**Individual pipelines**:
```bash
# Data validation
kedro run --pipeline=data_validation

# Data preprocessing
kedro run --pipeline=data_preprocessing

# Feature engineering
kedro run --pipeline=feature_engineering

# Model training
kedro run --pipeline=model_inference

# Model explainability
kedro run --pipeline=explainability

# Data drift detection
kedro run --pipeline=data_drift
```

**Training pipeline (recommended)**:
```bash
kedro run --pipeline=training_pipeline
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test
pytest tests/test_run.py -v
```

### MLflow UI

```bash
# Start MLflow tracking server
mlflow ui
```

## Pipeline Architecture

The project consists of 11 main pipelines that work together in sequence:

### 1. Data Units Tests (`data_units_test`)
- Validates raw input data using Great Expectations
- Ensures data quality and schema compliance

### 2. Data Preprocessing (`data_preprocessing`)
- Basic data cleaning operations
- Data type conversions and duplicate removal

### 3. Data Preparation (`data_preparation`)
- Target encoding and categorical encoding
- Date feature extraction and processing

### 4. Data Units Tests After Processing (`data_units_tests_after_processing`)
- Validates processed data using Great Expectations
- Ensures processed data meets quality standards

### 5. Data Split (`data_split`)
- Train/validation/test split with stratified sampling
- 90% temp_data, 10% test → 80% train, 20% validation

### 6. Data Transformations (`data_transformations`)
- Feature scaling (StandardScaler/MinMaxScaler)
- Missing value imputation (median/mean/KNN)
- Advanced processing from utils

### 7. Feature Engineering (`feature_engineering`)
- Date component extraction
- Interaction features and statistical features
- Polynomial features (configurable)

### 8. Feature Selection (`feature_selection`)
- XGBoost and Random Forest feature importance
- Combined selection strategies
- Feature ranking and selection

### 9. Model Inference (`model_inference`)
- Model training and evaluation
- MLflow experiment tracking
- Model performance metrics and artifacts

### 10. Explainability (`explainability`)
- Permutation importance analysis
- Model interpretability insights
- Feature importance validation

### 11. Data Drift (`data_drift`)
- Statistical drift detection between training and test data
- Kolmogorov-Smirnov test for numerical features
- Chi-square test for categorical features
- Population Stability Index (PSI) calculation
- Drift severity classification and reporting

## Complete Pipeline Flow

The training pipeline executes in the following sequence:

```
data_units_test → data_preprocessing → data_preparation → feature_engineering → data_units_tests_after_processing → data_split → data_transformations → feature_selection → model_inference → explainability → data_drift
```

## Configuration

### Parameters (`conf/base/parameters.yml`)
- Model hyperparameters for XGBoost and Random Forest
- Feature selection settings and thresholds
- Data processing parameters and feature lists
- Feature store configuration
- Explainability and data drift parameters

### Catalog (`conf/base/catalog.yml`)
- Data source definitions and storage locations
- MLflow artifact configurations
- Model registry settings
- Reporting outputs including drift reports and permutation importance

### MLflow (`conf/base/mlflow.yml`)
- Experiment tracking configuration
- Model registry settings
- Artifact storage configuration

## Model Performance

The current best model achieves:
- **Accuracy**: 87.3%
- **F1 Score**: 89.1%
- **Precision**: 88.5%
- **Recall**: 89.7%

## MLflow Integration

The project includes comprehensive MLflow integration:
- Automatic parameter logging for all models
- Metric tracking (F1 score, accuracy, precision, recall)
- Model versioning and artifact storage
- Feature importance logging and analysis
- Dataset information and statistics logging
- Prediction results and confusion matrices
- Permutation importance analysis and logging
- Data drift detection and reporting

## Model Interpretability

The explainability pipeline provides:
- **Permutation Importance**: Calculates feature importance by measuring performance drop when features are randomly shuffled
- **Feature Ranking**: Ranks features by their contribution to model performance
- **MLflow Integration**: Logs importance scores and creates artifacts for analysis
- **Statistical Validation**: Provides mean and standard deviation of importance scores

## Data Drift Detection

The data drift pipeline provides:
- **Statistical Analysis**: Comprehensive drift detection for numerical and categorical features
- **Distribution Comparison**: Kolmogorov-Smirnov test for numerical features and Chi-square test for categorical features
- **Population Stability Index**: PSI calculation for distribution stability assessment
- **Drift Severity Classification**: High/medium/low drift classification based on statistical significance
- **Overall Drift Score**: Percentage of features showing significant drift
- **Detailed Reporting**: Comprehensive drift reports with feature-level analysis

## Great Expectations Setup

### Raw Data Suite (`gx/expectations/raw_data_suite.json`)
- Validates original data schema
- Checks row count expectations
- Ensures column presence and data types

### Processed Data Suite (`gx/expectations/processed_data_suite.json`)
- Validates processed data quality
- Checks target column presence
- Validates data ranges and distributions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## Testing

The project includes comprehensive tests:
- Pipeline structure tests
- Data validation tests
- Model training tests
- Integration tests

Run tests with:
```bash
pytest tests/ -v
```

## Troubleshooting

### Common Issues

1. **MLflow nested run errors**: Ensure `nested=True` is set for nested runs
2. **Import errors**: Check that all dependencies are installed
3. **Data path issues**: Verify data files are in the correct locations
4. **Memory issues**: Consider reducing batch sizes or using data sampling

### Getting Help

- Check the logs in `logs/` directory
- Review MLflow experiment tracking
- Run tests to identify issues
- Check configuration files

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built with [Kedro](https://kedro.org/)
- ML experiment tracking with [MLflow](https://mlflow.org/)
- Data validation with [Great Expectations](https://greatexpectations.io/)
- Visualization with [Streamlit](https://streamlit.io/)
