# MLFlow in Claim Injury Prediction

## Overview

This project implements a comprehensive machine learning pipeline for predicting injury outcomes from insurance claims data using Kedro and MLflow. The system includes data validation, feature engineering, model training, and deployment capabilities.

## Project Structure

```
MLFlow-in-Claim-Injury-Prediction/
├── conf/                          # Configuration files
│   ├── base/                      # Base configuration
│   └── local/                     # Local configuration (not in git)
├── data/                          # Data files
│   ├── 01_raw/                    # Raw data
│   ├── 02_intermediate/           # Intermediate data
│   ├── 03_primary/                # Primary data
│   ├── 04_feature/                # Feature data
│   ├── 05_model_input/            # Model input data
│   ├── 06_models/                 # Trained models
│   └── 08_reporting/              # Reports and outputs
├── docs/                          # Documentation
├── notebooks/                     # Jupyter notebooks
├── src/                           # Source code
│   └── mlflow_in_claim_injury_prediction/
│       ├── pipelines/             # Kedro pipelines
│       └── utils/                 # Utility functions
├── tests/                         # Tests
├── streamlit/                     # Streamlit dashboard
└── utils/                         # Utility modules
```

## Key Features

- **Data Validation**: Great Expectations integration for data quality checks
- **Feature Engineering**: Advanced feature creation and selection
- **Model Training**: XGBoost and Random Forest with MLflow tracking
- **Model Deployment**: MLflow model registry integration
- **Monitoring**: Data drift detection and model performance tracking
- **Dashboard**: Streamlit-based visualization and prediction interface

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

**Full training pipeline**:
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
```

**From specific nodes**:
```bash
kedro run --from-nodes=train_and_evaluate_models_node
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

### Streamlit Dashboard

```bash
# Start the dashboard
streamlit run streamlit/00_home_page.py
```

### MLflow UI

```bash
# Start MLflow tracking server
mlflow ui
```

## Pipeline Architecture

### 1. Data Validation (`data_validation`)
- Raw data quality checks using Great Expectations
- Data schema validation
- Missing value detection

### 2. Data Preprocessing (`data_preprocessing`)
- Data type conversions
- Basic cleaning operations
- Duplicate removal

### 3. Data Preparation (`data_preparation`)
- Target encoding
- Categorical encoding
- Date feature extraction

### 4. Feature Engineering (`feature_engineering`)
- Advanced feature creation
- Interaction features
- Statistical features

### 5. Data Split (`data_split`)
- Train/validation/test split
- Stratified sampling
- Data versioning

### 6. Data Transformations (`data_transformations`)
- Feature scaling
- Missing value imputation
- Feature creation

### 7. Feature Selection (`feature_selection`)
- XGBoost-based feature selection
- Random Forest feature importance
- Feature ranking and selection

### 8. Model Inference (`model_inference`)
- Model training and evaluation
- MLflow experiment tracking
- Model performance metrics

## Configuration

### Parameters (`conf/base/parameters.yml`)
- Model hyperparameters
- Feature selection settings
- Data processing parameters

### Catalog (`conf/base/catalog.yml`)
- Data source definitions
- MLflow artifact configurations
- Model registry settings

### MLflow (`conf/local/mlflow.yml`)
- Experiment tracking configuration
- Model registry settings
- Artifact storage configuration

## Model Performance

The current best model achieves:
- **Accuracy**: 87.3%
- **F1 Score**: 89.1%
- **Precision**: 88.5%
- **Recall**: 89.7%

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

## Deployment

### Docker Deployment
```bash
# Build the Docker image
docker build -t claim-injury-prediction .

# Run the container
docker run -p 8501:8501 claim-injury-prediction
```

### MLflow Model Serving
```bash
# Serve the best model
mlflow models serve -m runs:/<run_id>/model -p 5001
```

## Monitoring

- **Data Drift**: Automated detection using statistical tests
- **Model Performance**: Continuous monitoring via MLflow
- **Data Quality**: Great Expectations validation suites
- **Pipeline Health**: Kedro pipeline monitoring

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

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Kedro](https://kedro.org/)
- ML experiment tracking with [MLflow](https://mlflow.org/)
- Data validation with [Great Expectations](https://greatexpectations.io/)
- Visualization with [Streamlit](https://streamlit.io/)
