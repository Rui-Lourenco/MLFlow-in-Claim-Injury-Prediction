# MLFlow Claim Injury Prediction Pipeline

This document describes the Kedro pipeline structure for the MLFlow Claim Injury Prediction project.

## Pipeline Overview

The project consists of 11 main pipelines that work together to process data, engineer features, select the best features, train models, provide interpretability, detect data drift, and perform inference with comprehensive MLFlow integration.

## Pipeline Structure

### 1. Data Units Tests (`data_units_test`)
- **Purpose**: Validates raw input data using Great Expectations
- **Input**: `raw_input_data`
- **Output**: `raw_data_validated`
- **Validation**: Ensures data quality and schema compliance

### 2. Data Preprocessing (`data_preprocessing`)
- **Purpose**: Basic data cleaning and type conversions
- **Input**: `raw_data_validated`
- **Output**: `processed_data`
- **Processing**: Data type conversions, duplicate removal, basic cleaning

### 3. Data Preparation (`data_preparation`)
- **Purpose**: Target encoding and categorical processing
- **Input**: `processed_data`
- **Output**: `prepared_data`
- **Processing**: Target encoding, categorical encoding, date feature extraction

### 4. Data Units Tests After Processing (`data_units_tests_after_processing`)
- **Purpose**: Validates processed data using Great Expectations
- **Input**: `prepared_data`
- **Output**: `prepared_data_validated`
- **Validation**: Ensures processed data meets quality standards

### 5. Data Split (`data_split`)
- **Purpose**: Splits data into train, validation, and test sets
- **Input**: `prepared_data_validated`
- **Output**: `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`
- **Split Strategy**: 
  - First split: 90% temp_data, 10% test
  - Second split: 80% train, 20% validation (from temp_data)

### 6. Data Transformations (`data_transformations`)
- **Purpose**: Handles missing values and scales numerical features
- **Input**: `X_train`, `X_val`, `X_test`
- **Output**: `X_train_final`, `X_val_final`, `X_test_final`, `scaler`, `imputer`
- **Processing**:
  - NA imputation using median/mean/KNN
  - Feature scaling (StandardScaler/MinMaxScaler)
  - Advanced processing from utils (frequency encoding, new features)

### 7. Feature Engineering (`feature_engineering`)
- **Purpose**: Creates new features from existing data
- **Input**: `X_train_final`, `X_val_final`, `X_test_final`
- **Output**: `X_train_engineered`, `X_val_engineered`, `X_test_engineered`, `feature_names`
- **Engineering**:
  - Date component extraction
  - Interaction features
  - Polynomial features (optional)
  - Statistical features

### 8. Feature Selection (`feature_selection`)
- **Purpose**: Selects the most important features using XGBoost and Random Forest
- **Input**: `X_train_engineered`, `X_val_engineered`, `X_test_engineered`, `y_train`, `y_val`
- **Output**: `X_train_selected`, `X_val_selected`, `X_test_selected`, `selected_features`, `feature_importance_summary`
- **Selection Methods**:
  - XGBoost feature importance
  - Random Forest feature importance
  - Combined selection (union/intersection)

### 9. Model Inference (`model_inference`)
- **Purpose**: Trains models and performs inference with MLFlow tracking
- **Input**: `X_train_selected`, `X_val_selected`, `X_test_selected`, `y_train`, `y_val`, `y_test`
- **Output**: Models, predictions, evaluation metrics
- **Models**: XGBoost and Random Forest
- **MLFlow Integration**: Parameter logging, metric tracking, model versioning

### 10. Explainability (`explainability`)
- **Purpose**: Provides model interpretability through permutation importance
- **Input**: `best_model`, `X_train_selected`, `y_train`
- **Output**: `permutation_importance`
- **Analysis**:
  - Permutation importance calculation
  - Feature ranking by importance
  - Statistical validation of feature contributions

### 11. Data Drift (`data_drift`)
- **Purpose**: Detects data drift between training and test datasets
- **Input**: `X_train_selected`, `X_test_selected`, `params:numerical_features`, `params:categorical_features`
- **Output**: `data_drift_report`
- **Analysis**:
  - Statistical drift detection using KS test for numerical features
  - Chi-square test for categorical features
  - Distribution drift using Population Stability Index (PSI)
  - Overall drift score calculation
  - Drift severity classification (high/medium/low)

## Complete Pipeline

The `training_pipeline` combines all individual pipelines in sequence:

```
data_units_test → data_preprocessing → data_preparation → feature_engineering → data_units_tests_after_processing → data_split → data_transformations → feature_engineering → feature_selection → model_inference → explainability → data_drift
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

## Usage

### Run Individual Pipelines
```bash
kedro run --pipeline=data_split
kedro run --pipeline=feature_engineering
kedro run --pipeline=model_inference
kedro run --pipeline=explainability
kedro run --pipeline=data_drift
```

### Run Complete Training Pipeline
```bash
kedro run --pipeline=training_pipeline
```

### Run Default Pipeline
```bash
kedro run
```

## Model Outputs

The pipeline produces:
- Trained models (XGBoost and Random Forest) with MLflow artifacts
- Feature importance rankings and analysis
- Test predictions and probabilities
- Evaluation metrics and classification reports
- Confusion matrices and detailed results
- Permutation importance analysis for model interpretability
- Data drift reports with statistical analysis and severity classification

## Pipeline Dependencies

### Data Flow Dependencies
- **model_inference** depends on feature selection outputs
- **explainability** depends on trained model from model_inference
- **data_drift** is independent and can run after data split

### Execution Order
The training pipeline executes in the following logical sequence:
1. Data validation and preprocessing
2. Feature engineering and selection
3. Model training and evaluation
4. Model interpretability analysis
5. Data drift detection

This order ensures that:
- All data quality checks are performed before processing
- Features are engineered and selected before model training
- Model interpretability analysis uses the trained model
- Data drift detection compares the final selected features between training and test sets 