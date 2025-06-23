# MLFlow Claim Injury Prediction Pipeline

This document describes the Kedro pipeline structure for the MLFlow Claim Injury Prediction project.

## Pipeline Overview

The project consists of 8 main pipelines that work together to process data, engineer features, select the best features, and train models with MLFlow integration.

## Pipeline Structure

### 1. Data Units Tests (`data_units_tests`)
- **Purpose**: Validates raw input data using Great Expectations
- **Input**: `raw_input_data`
- **Output**: `raw_data_validated`
- **Validation**: Ensures data quality and schema compliance

### 2. Data Units Tests After Processing (`data_units_tests_after_processing`)
- **Purpose**: Validates processed data using Great Expectations
- **Input**: `processed_data`
- **Output**: `processed_data_validated`
- **Validation**: Ensures processed data meets quality standards

### 3. Data Split (`data_split`)
- **Purpose**: Splits data into train, validation, and test sets
- **Input**: `processed_data_validated`
- **Output**: `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`
- **Split Strategy**: 
  - First split: 90% temp_data, 10% test
  - Second split: 80% train, 20% validation (from temp_data)

### 4. Data Processing (`data_processing`)
- **Purpose**: Handles missing values and scales numerical features
- **Input**: `X_train`, `X_val`, `X_test`
- **Output**: `X_train_final`, `X_val_final`, `X_test_final`, `scaler`, `imputer`
- **Processing**:
  - NA imputation using median/mean/KNN
  - Feature scaling (StandardScaler/MinMaxScaler)
  - Advanced processing from utils (frequency encoding, new features)

### 5. Feature Engineering (`feature_engineering`)
- **Purpose**: Creates new features from existing data
- **Input**: `X_train_final`, `X_val_final`, `X_test_final`
- **Output**: `X_train_engineered`, `X_val_engineered`, `X_test_engineered`, `feature_names`
- **Engineering**:
  - Date component extraction
  - Interaction features
  - Polynomial features (optional)
  - Statistical features

### 6. Feature Selection (`feature_selection`)
- **Purpose**: Selects the most important features using XGBoost and Random Forest
- **Input**: `X_train_engineered`, `X_val_engineered`, `X_test_engineered`, `y_train`, `y_val`
- **Output**: `X_train_selected`, `X_val_selected`, `X_test_selected`, `selected_features`, `feature_importance_summary`
- **Selection Methods**:
  - XGBoost feature importance
  - Random Forest feature importance
  - Combined selection (union/intersection)

### 7. Model Inference (`model_inference`)
- **Purpose**: Trains models and performs inference with MLFlow tracking
- **Input**: `X_train_selected`, `X_val_selected`, `X_test_selected`, `y_train`, `y_val`, `y_test`
- **Output**: Models, predictions, evaluation metrics
- **Models**: XGBoost and Random Forest
- **MLFlow Integration**: Parameter logging, metric tracking, model versioning

## Complete Pipeline

The `complete_pipeline` combines all individual pipelines in sequence:

```
data_units_tests → data_units_tests_after_processing → data_split → data_processing → feature_engineering → feature_selection → model_inference
```

## Key Features

### MLFlow Integration
- Automatic parameter logging
- Metric tracking (F1 score, accuracy, precision, recall)
- Model versioning and artifact storage
- Feature importance logging

### Great Expectations
- Raw data validation
- Processed data validation
- Data quality monitoring

### Feature Engineering
- Date component extraction
- Interaction features
- Statistical features
- Polynomial features (configurable)

### Feature Selection
- XGBoost-based selection
- Random Forest-based selection
- Combined selection strategies
- Feature importance analysis

### Model Training
- XGBoost classifier
- Random Forest classifier
- Automatic best model selection
- Comprehensive evaluation metrics

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
- Data split ratios
- Feature lists (numerical/categorical)
- Processing methods
- Model hyperparameters
- Feature selection thresholds

### `conf/base/mlflow.yml`
MLFlow configuration for parameter handling.

## Great Expectations Setup

### Raw Data Suite (`gx/expectations/raw_data_suite.json`)
- Validates original data schema
- Checks row count expectations
- Ensures column presence

### Processed Data Suite (`gx/expectations/processed_data_suite.json`)
- Validates processed data quality
- Checks target column presence
- Validates data ranges

## Usage

### Run Individual Pipelines
```bash
kedro run --pipeline=data_split
kedro run --pipeline=feature_engineering
kedro run --pipeline=model_inference
```

### Run Complete Pipeline
```bash
kedro run --pipeline=complete_pipeline
```

## Model Outputs

The pipeline produces:
- Trained models (XGBoost and Random Forest)
- Feature importance rankings
- Test predictions and probabilities
- Evaluation metrics and reports
- Confusion matrices

## Data Flow

1. **Raw Data** → Validation → **Validated Raw Data**
2. **Validated Raw Data** → Processing → **Processed Data**
3. **Processed Data** → Validation → **Validated Processed Data**
4. **Validated Processed Data** → Split → **Train/Val/Test Sets**
5. **Train/Val/Test Sets** → Processing → **Final Processed Sets**
6. **Final Processed Sets** → Engineering → **Engineered Features**
7. **Engineered Features** → Selection → **Selected Features**
8. **Selected Features** → Training → **Models & Predictions**

## Dependencies

The pipeline leverages existing utilities:
- `utils/utils.py`: Core data processing functions
- `utils/utils_feature_selection.py`: Feature selection utilities
- `utils/utils_dicts.py`: Feature definitions
- `utils/model.py`: Model training and inference functions

## MLFlow Artifacts

The pipeline automatically logs:
- Model parameters
- Training and validation metrics
- Feature importance files
- Classification reports
- Confusion matrices
- Prediction results

This comprehensive pipeline structure ensures reproducible, monitored, and scalable machine learning workflows for claim injury prediction. 