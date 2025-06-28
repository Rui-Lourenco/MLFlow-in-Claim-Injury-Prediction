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
│       │   ├── data_units_tests_after_processing/ # Processed data validation
│       │   ├── feature_store/     # Feature store integration
│       │   └── visualization/     # Data visualization
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
- **Feature Store Integration**: Hopsworks feature store for feature management and versioning
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

# Data preparation
kedro run --pipeline=data_preparation

# Feature engineering
kedro run --pipeline=feature_engineering

# Feature selection
kedro run --pipeline=feature_selection

# Model training
kedro run --pipeline=model_inference

# Model explainability
kedro run --pipeline=explainability

# Data drift detection
kedro run --pipeline=data_drift

# Feature store operations
kedro run --pipeline=feature_store

# Visualization
kedro run --pipeline=visualization
```

**Training pipeline (recommended)**:
```bash
kedro run --pipeline=training_pipeline
```

**Inference pipeline**:
```bash
kedro run --pipeline=inference_pipeline
```

**Reporting pipeline**:
```bash
kedro run --pipeline=reporting_pipeline
```

### MLflow UI

```bash
# Start MLflow tracking server
kedro mlflow ui
```

## Pipeline Architecture

The project consists of 13 main pipelines that work together in sequence:

### 1. Data Units Tests (`data_units_test`)
- Validates raw input data using Great Expectations
- Ensures data quality and schema compliance

### 2. Data Preprocessing (`data_preprocessing`)
- Basic data cleaning operations
- Data type conversions and duplicate removal

### 3. Data Preparation (`data_preparation`)
- Target encoding and categorical encoding
- Date feature extraction and processing

### 4. Feature Engineering (`feature_engineering`)
- Date component extraction
- Interaction features and statistical features
- Polynomial features (configurable)

### 5. Data Units Tests After Processing (`data_units_tests_after_processing`)
- Validates processed data using Great Expectations
- Ensures processed data meets quality standards

### 6. Data Split (`data_split`)
- Train/validation/test split with stratified sampling
- 90% temp_data, 10% test → 80% train, 20% validation

### 7. Data Transformations (`data_transformations`)
- Feature scaling (StandardScaler/MinMaxScaler)
- Missing value imputation (median/mean/KNN)
- Advanced processing from utils

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

### 12. Feature Store (`feature_store`)
- Feature store integration with Hopsworks
- Feature group management and versioning
- Feature validation and metadata tracking

### 13. Visualization (`visualization`)
- Data visualization and reporting
- Model performance charts
- Feature importance plots

## Complete Pipeline Flow

The `training_pipeline` combines all individual pipelines in sequence:

```
data_units_test → data_preprocessing → data_preparation → feature_engineering → data_units_tests_after_processing → data_split → data_transformations → feature_selection → model_inference → explainability → data_drift → visualization
```

## Key Features

### MLFlow Integration
- Automatic parameter logging for all models
- Metric tracking (F1 score, accuracy, precision, recall)
- Model versioning and artifact storage
- Feature importance logging and analysis
- Dataset information and statistics logging
- Prediction results and confusion matrices
- Permutation importance analysis and logging
- Data drift detection and reporting

### Great Expectations
- Raw data validation with schema checks
- Processed data validation with quality standards
- Data quality monitoring and reporting

### Feature Engineering
- Date component extraction (year, month, day, dayofweek)
- Interaction features between key variables
- Statistical features and aggregations
- Polynomial features (configurable)

### Feature Selection
- XGBoost-based feature importance ranking
- Random Forest-based feature importance ranking
- Combined selection strategies (union/intersection)
- Feature importance analysis and visualization

### Model Training
- XGBoost classifier with optimized parameters
- Random Forest classifier with optimized parameters
- Automatic best model selection based on validation performance
- Comprehensive evaluation metrics and reporting

### Model Interpretability
- Permutation importance analysis for feature understanding
- Statistical validation of feature contributions
- MLflow integration for importance tracking
- Feature ranking and visualization

### Data Drift Detection
- Comprehensive statistical analysis for numerical and categorical features
- Kolmogorov-Smirnov test for distribution differences
- Chi-square test for categorical feature drift
- Population Stability Index (PSI) calculation
- Drift severity classification and reporting
- Overall drift score calculation

### Feature Store Integration
- Hopsworks feature store for feature management
- Feature group versioning and metadata tracking
- Feature validation and quality monitoring
- Automated feature pipeline integration

## Configuration Files

### `conf/base/catalog.yml`
Defines all data products and their storage locations:
- Raw data: `data/01_raw/`
- Intermediate data: `data/02_intermediate/`
- Primary data: `data/03_primary/`
- Model input: `data/04_model_input/`
- Encoders: `data/05_encoders/`
- Models: `data/06_models/`
- Model output: `data/07_model_output/`
- Reporting: `data/08_reporting/`

### `conf/base/parameters.yml`
Contains all pipeline parameters:
- Data split ratios and random state
- Feature lists (numerical/categorical)
- Processing methods and scaling options
- Model hyperparameters for XGBoost and Random Forest
- Feature selection thresholds and methods
- Feature store configuration
- Explainability parameters (n_repeats, random_state)
- Data drift detection parameters

### `conf/base/mlflow.yml`
MLFlow configuration for experiment tracking and model registry.

## Great Expectations Setup

### Raw Data Suite (`gx/expectations/raw_data_suite.json`)
- Validates original data schema and structure
- Checks row count expectations
- Ensures column presence and data types

### Processed Data Suite (`gx/expectations/processed_data_suite.json`)
- Validates processed data quality and completeness
- Checks target column presence and distribution
- Validates data ranges and statistical properties

## Acknowledgments

- Built with [Kedro](https://kedro.org/)
- ML experiment tracking with [MLflow](https://mlflow.org/)
- Data validation with [Great Expectations](https://greatexpectations.io/)
