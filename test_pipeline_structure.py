#!/usr/bin/env python3
"""
Test script to verify the Kedro pipeline structure.
This script checks that all pipelines can be imported and have the correct structure.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

def test_pipeline_imports():
    """Test that all pipeline modules can be imported."""
    print("Testing pipeline imports...")
    
    try:
        from mlflow_in_claim_injury_prediction.pipeline_registry import register_pipelines
        print("✓ Pipeline registry imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pipeline registry: {e}")
        return False
    
    try:
        # Test individual pipeline imports
        from mlflow_in_claim_injury_prediction.pipelines import data_units_test
        print("✓ data_units_test pipeline imported")
        
        from mlflow_in_claim_injury_prediction.pipelines import data_units_tests_after_processing
        print("✓ data_units_tests_after_processing pipeline imported")
        
        from mlflow_in_claim_injury_prediction.pipelines import data_preparation
        print("✓ data_preparation pipeline imported")
        
        from mlflow_in_claim_injury_prediction.pipelines import data_split
        print("✓ data_split pipeline imported")
        
        from mlflow_in_claim_injury_prediction.pipelines import data_processing
        print("✓ data_processing pipeline imported")
        
        from mlflow_in_claim_injury_prediction.pipelines import feature_engineering
        print("✓ feature_engineering pipeline imported")
        
        from mlflow_in_claim_injury_prediction.pipelines import feature_selection
        print("✓ feature_selection pipeline imported")
        
        from mlflow_in_claim_injury_prediction.pipelines import feature_store
        print("✓ feature_store pipeline imported")
        
        from mlflow_in_claim_injury_prediction.pipelines import data_upload
        print("✓ data_upload pipeline imported")
        
        from mlflow_in_claim_injury_prediction.pipelines import data_drift
        print("✓ data_drift pipeline imported")
        
        from mlflow_in_claim_injury_prediction.pipelines import model_inference
        print("✓ model_inference pipeline imported")
        
        print("✓ All pipeline modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pipeline modules: {e}")
        return False
    
    return True

def test_pipeline_creation():
    """Test that all pipelines can be created."""
    print("\nTesting pipeline creation...")
    
    try:
        from mlflow_in_claim_injury_prediction.pipeline_registry import register_pipelines
        
        pipelines = register_pipelines()
        
        expected_pipelines = [
            "data_units_tests",
            "data_units_tests_after_processing",
            "data_preparation",
            "data_split",
            "data_processing",
            "feature_engineering",
            "feature_selection",
            "feature_store",
            "data_upload",
            "data_drift",
            "model_inference",
            "complete_pipeline"
        ]
        
        for pipeline_name in expected_pipelines:
            if pipeline_name in pipelines:
                print(f"✓ Pipeline '{pipeline_name}' created successfully")
            else:
                print(f"✗ Pipeline '{pipeline_name}' not found")
                return False
        
        print(f"✓ All {len(expected_pipelines)} pipelines created successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to create pipelines: {e}")
        return False

def test_utils_imports():
    """Test that utility modules can be imported."""
    print("\nTesting utility imports...")
    
    try:
        # Test individual utility imports
        import utils.utils
        print("✓ utils.py imported successfully")
        
        import utils.utils_feature_selection
        print("✓ utils_feature_selection.py imported successfully")
        
        import utils.utils_dicts
        print("✓ utils_dicts.py imported successfully")
        
        import utils.model
        print("✓ model.py imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import utility modules: {e}")
        return False

def test_configuration_files():
    """Test that configuration files exist and are valid."""
    print("\nTesting configuration files...")
    
    config_files = [
        "conf/base/catalog.yml",
        "conf/base/parameters.yml", 
        "conf/base/mlflow.yml",
        "gx/great_expectations.yml",
        "gx/expectations/raw_data_suite.json",
        "gx/expectations/processed_data_suite.json"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✓ {config_file} exists")
        else:
            print(f"✗ {config_file} not found")
            return False
    
    return True

def test_directory_structure():
    """Test that the required directory structure exists."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        "src/mlflow_in_claim_injury_prediction/pipelines/data_units_tests_after_processing",
        "src/mlflow_in_claim_injury_prediction/pipelines/data_preparation",
        "src/mlflow_in_claim_injury_prediction/pipelines/data_split",
        "src/mlflow_in_claim_injury_prediction/pipelines/data_processing",
        "src/mlflow_in_claim_injury_prediction/pipelines/feature_engineering",
        "src/mlflow_in_claim_injury_prediction/pipelines/feature_selection",
        "src/mlflow_in_claim_injury_prediction/pipelines/feature_store",
        "src/mlflow_in_claim_injury_prediction/pipelines/data_upload",
        "src/mlflow_in_claim_injury_prediction/pipelines/data_drift",
        "src/mlflow_in_claim_injury_prediction/pipelines/model_inference"
    ]
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ {directory} exists")
        else:
            print(f"✗ {directory} not found")
            return False
    
    return True

def test_pipeline_files():
    """Test that pipeline files exist."""
    print("\nTesting pipeline files...")
    
    pipeline_files = [
        "src/mlflow_in_claim_injury_prediction/pipelines/data_units_tests_after_processing/__init__.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/data_units_tests_after_processing/nodes.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/data_units_tests_after_processing/pipeline.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/data_preparation/__init__.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/data_preparation/nodes.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/data_preparation/pipeline.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/data_split/__init__.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/data_split/nodes.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/data_split/pipeline.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/data_processing/__init__.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/data_processing/nodes.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/data_processing/pipeline.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/feature_engineering/__init__.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/feature_engineering/nodes.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/feature_engineering/pipeline.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/feature_selection/__init__.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/feature_selection/nodes.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/feature_selection/pipeline.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/feature_store/__init__.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/feature_store/nodes.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/feature_store/pipeline.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/data_upload/__init__.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/data_upload/nodes.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/data_upload/pipeline.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/data_upload/utils.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/data_drift/__init__.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/data_drift/nodes.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/data_drift/pipeline.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/model_inference/__init__.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/model_inference/nodes.py",
        "src/mlflow_in_claim_injury_prediction/pipelines/model_inference/pipeline.py"
    ]
    
    for file_path in pipeline_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} not found")
            return False
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Kedro Pipeline Structure Test")
    print("=" * 60)
    
    tests = [
        test_directory_structure,
        test_pipeline_files,
        test_configuration_files,
        test_utils_imports,
        test_pipeline_imports,
        test_pipeline_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ Test {test.__name__} failed")
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The pipeline structure is ready.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 