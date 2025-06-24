# MLflow Implementation Improvements

## Issues Fixed

### 1. Gaudy Run Names (e.g., "gaudy-conch-754")
**Problem**: MLflow was using auto-generated random names instead of descriptive names.

**Solution**: 
- Created `create_experiment_run_name()` function that generates descriptive names with timestamps
- Examples: `model_training_xgboost_20250624_020937`, `data_split_20250624_020938`
- Added proper experiment setup with `setup_mlflow()` function

### 2. Missing Dataset and Model Artifacts
**Problem**: Datasets and models were showing as "-" in the MLflow UI.

**Solution**:
- Implemented comprehensive artifact logging with `log_dataset_info()`, `log_model_with_metadata()`
- Added proper model logging for both XGBoost and Random Forest models
- Created detailed feature importance logging with `log_feature_importance()`
- Added prediction logging with `log_predictions_with_metadata()`

### 3. Short Duration and Missing Metrics
**Problem**: Runs appeared to have very short duration and limited metrics.

**Solution**:
- Added comprehensive metrics logging for all pipeline steps
- Implemented proper run timing with context managers
- Added detailed parameter logging for all models and preprocessing steps
- Created structured logging for training, validation, and test metrics

## New MLflow Utilities

### Core Functions

#### `setup_mlflow(experiment_name)`
- Sets up MLflow tracking URI to local filesystem
- Creates and sets the experiment
- Ensures consistent experiment naming

#### `start_mlflow_run(run_name, tags)`
- Starts MLflow runs with proper naming and tags
- Adds default project tags and timestamps
- Returns active run context manager

#### `create_experiment_run_name(pipeline_step, model_name)`
- Generates descriptive run names with timestamps
- Examples: `data_split_20250624_020938`, `model_training_xgboost_20250624_020937`

### Logging Functions

#### `log_dataset_info(data, dataset_name, description)`
- Logs dataset shape, columns, and rows
- Creates dataset statistics and missing value analysis
- Saves dataset samples as artifacts

#### `log_model_with_metadata(model, model_name, model_type, feature_names, model_params)`
- Logs model parameters and metadata
- Saves models using appropriate MLflow model logging
- Tracks feature names and model configuration

#### `log_feature_importance(feature_importance, model_name)`
- Logs feature importance with proper formatting
- Creates top features and all features artifacts
- Tracks feature importance statistics

#### `log_predictions_with_metadata(y_true, y_pred, y_proba, dataset_name)`
- Logs predictions with confidence scores
- Creates detailed prediction artifacts
- Tracks prediction statistics

#### `log_model_evaluation_metrics(metrics, dataset_name)`
- Logs evaluation metrics for train/val/test sets
- Supports multiple metric types (accuracy, f1, precision, recall)

## Updated Configuration

### `conf/base/mlflow.yml`
```yaml
tracking:
  params:
    long_params_strategy: tag
  experiment:
    name: "claim_injury_prediction"
    artifact_location: "mlruns"
  run:
    name: "claim_injury_model_training"
    tags:
      project: "claim_injury_prediction"
      version: "1.0"
  artifacts:
    log_datasets: true
    log_models: true
    log_feature_importance: true
    log_predictions: true
    log_confusion_matrix: true
    log_classification_report: true
```

## Updated Pipeline Nodes

### Model Training (`utils/model.py`)
- **XGBoost Training**: Now logs comprehensive metrics, feature importance, and model artifacts
- **Random Forest Training**: Same comprehensive logging as XGBoost
- **Model Selection**: Logs best model selection with validation metrics

### Data Split (`pipelines/data_split/nodes.py`)
- Logs original dataset information
- Tracks data split percentages and class balance
- Saves target distributions for each split

### Model Inference (`pipelines/model_inference/nodes.py`)
- Logs complete training pipeline with all datasets
- Tracks best model selection and test evaluation
- Generates detailed prediction artifacts

## Artifacts Now Logged

### Datasets
- `train_features_sample.csv` - Training data sample
- `val_features_sample.csv` - Validation data sample  
- `test_features_sample.csv` - Test data sample
- `train_target_distribution.csv` - Target distribution
- `val_target_distribution.csv` - Validation target distribution
- `test_target_distribution.csv` - Test target distribution

### Models
- `xgboost_model` - XGBoost model artifact
- `random_forest_model` - Random Forest model artifact
- `best_xgboost_model` / `best_random_forest_model` - Best model artifact

### Feature Importance
- `xgboost_top_features.csv` - Top 20 XGBoost features
- `xgboost_all_features.csv` - All XGBoost feature importance
- `random_forest_top_features.csv` - Top 20 Random Forest features
- `random_forest_all_features.csv` - All Random Forest feature importance

### Predictions
- `train_predictions.csv` - Training predictions with probabilities
- `val_predictions.csv` - Validation predictions with probabilities
- `test_predictions.csv` - Test predictions with probabilities
- `detailed_predictions.csv` - Complete prediction results

### Evaluation
- `train_classification_report.csv` - Training classification report
- `val_classification_report.csv` - Validation classification report
- `test_classification_report.csv` - Test classification report
- `confusion_matrix.csv` - Confusion matrix

## Metrics Now Tracked

### Data Split Metrics
- `train_samples`, `val_samples`, `test_samples`
- `train_percentage`, `val_percentage`, `test_percentage`
- `train_class_balance`, `val_class_balance`, `test_class_balance`

### Model Training Metrics
- `train_f1_score`, `val_f1_score`, `test_f1_score`
- `train_accuracy`, `val_accuracy`, `test_accuracy`
- `train_precision`, `val_precision`, `test_precision`
- `train_recall`, `val_recall`, `test_recall`

### Feature Importance Metrics
- `total_features`, `max_importance`, `min_importance`
- `mean_importance`, `std_importance`

### Prediction Metrics
- `total_predictions`, `unique_predictions`
- `avg_confidence`, `min_confidence`, `max_confidence`

## Testing

Run the test script to verify the implementation:
```bash
python test_mlflow_implementation.py
```

Start the MLflow UI to view results:
```bash
mlflow ui --port 5000
```

## Expected Results

After these improvements, you should see:

1. **Descriptive Run Names**: Instead of "gaudy-conch-754", you'll see names like "model_training_xgboost_20250624_020937"

2. **Proper Duration**: Runs will show actual execution time instead of appearing very short

3. **Rich Artifacts**: Datasets and models will be properly logged and visible in the UI

4. **Comprehensive Metrics**: All training, validation, and test metrics will be tracked

5. **Organized Experiments**: All runs will be organized under the "claim_injury_prediction" experiment

## Next Steps

1. Run the complete pipeline: `kedro run --pipeline=model_inference`
2. Check the MLflow UI at http://localhost:5000
3. Verify that run names are descriptive and all artifacts are properly logged
4. Review the comprehensive metrics and feature importance analysis 