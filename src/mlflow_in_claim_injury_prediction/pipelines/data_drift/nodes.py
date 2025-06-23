import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json

log = logging.getLogger(__name__)

def calculate_statistical_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str]
) -> Dict[str, Any]:
    """
    Calculate statistical drift between reference and current data.
    
    Args:
        reference_data: Reference dataset (training data)
        current_data: Current dataset (new data)
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        
    Returns:
        Dictionary containing drift statistics for each feature
    """
    
    log.info("Calculating statistical drift between reference and current data")
    
    drift_results = {
        "numerical_drift": {},
        "categorical_drift": {},
        "overall_drift_score": 0.0,
        "drift_summary": {}
    }
    
    # Calculate drift for numerical features using KS test
    for feature in numerical_features:
        if feature in reference_data.columns and feature in current_data.columns:
            try:
                # Remove NaN values for statistical tests
                ref_values = reference_data[feature].dropna()
                cur_values = current_data[feature].dropna()
                
                if len(ref_values) > 0 and len(cur_values) > 0:
                    # Perform Kolmogorov-Smirnov test
                    ks_statistic, p_value = ks_2samp(ref_values, cur_values)
                    
                    # Calculate distribution statistics
                    ref_mean, ref_std = ref_values.mean(), ref_values.std()
                    cur_mean, cur_std = cur_values.mean(), cur_values.std()
                    
                    # Calculate drift magnitude
                    mean_drift = abs(cur_mean - ref_mean) / (ref_std + 1e-8)
                    std_drift = abs(cur_std - ref_std) / (ref_std + 1e-8)
                    
                    drift_results["numerical_drift"][feature] = {
                        "ks_statistic": ks_statistic,
                        "p_value": p_value,
                        "drift_detected": p_value < 0.05,
                        "reference_mean": ref_mean,
                        "current_mean": cur_mean,
                        "reference_std": ref_std,
                        "current_std": cur_std,
                        "mean_drift_magnitude": mean_drift,
                        "std_drift_magnitude": std_drift,
                        "drift_severity": "high" if p_value < 0.01 else "medium" if p_value < 0.05 else "low"
                    }
                    
                    log.info(f"Numerical drift for {feature}: p-value={p_value:.4f}, drift_detected={p_value < 0.05}")
                    
            except Exception as e:
                log.warning(f"Error calculating drift for numerical feature {feature}: {e}")
                drift_results["numerical_drift"][feature] = {"error": str(e)}
    
    # Calculate drift for categorical features using Chi-square test
    for feature in categorical_features:
        if feature in reference_data.columns and feature in current_data.columns:
            try:
                # Create contingency table
                ref_counts = reference_data[feature].value_counts()
                cur_counts = current_data[feature].value_counts()
                
                # Align categories
                all_categories = set(ref_counts.index) | set(cur_counts.index)
                ref_aligned = ref_counts.reindex(all_categories, fill_value=0)
                cur_aligned = cur_counts.reindex(all_categories, fill_value=0)
                
                # Perform Chi-square test
                contingency_table = np.array([ref_aligned.values, cur_aligned.values])
                chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                
                # Calculate distribution differences
                ref_props = ref_aligned / ref_aligned.sum()
                cur_props = cur_aligned / cur_aligned.sum()
                max_prop_diff = (cur_props - ref_props).abs().max()
                
                drift_results["categorical_drift"][feature] = {
                    "chi2_statistic": chi2_stat,
                    "p_value": p_value,
                    "drift_detected": p_value < 0.05,
                    "reference_proportions": ref_props.to_dict(),
                    "current_proportions": cur_props.to_dict(),
                    "max_proportion_difference": max_prop_diff,
                    "drift_severity": "high" if p_value < 0.01 else "medium" if p_value < 0.05 else "low"
                }
                
                log.info(f"Categorical drift for {feature}: p-value={p_value:.4f}, drift_detected={p_value < 0.05}")
                
            except Exception as e:
                log.warning(f"Error calculating drift for categorical feature {feature}: {e}")
                drift_results["categorical_drift"][feature] = {"error": str(e)}
    
    # Calculate overall drift score
    total_features = len(numerical_features) + len(categorical_features)
    drifted_features = 0
    
    for feature_data in drift_results["numerical_drift"].values():
        if isinstance(feature_data, dict) and "drift_detected" in feature_data:
            if feature_data["drift_detected"]:
                drifted_features += 1
    
    for feature_data in drift_results["categorical_drift"].values():
        if isinstance(feature_data, dict) and "drift_detected" in feature_data:
            if feature_data["drift_detected"]:
                drifted_features += 1
    
    drift_results["overall_drift_score"] = drifted_features / total_features if total_features > 0 else 0.0
    
    # Create drift summary
    drift_results["drift_summary"] = {
        "total_features": total_features,
        "drifted_features": drifted_features,
        "drift_percentage": drift_results["overall_drift_score"] * 100,
        "high_drift_features": [],
        "medium_drift_features": [],
        "low_drift_features": []
    }
    
    # Categorize features by drift severity
    for feature, data in drift_results["numerical_drift"].items():
        if isinstance(data, dict) and "drift_severity" in data:
            if data["drift_severity"] == "high":
                drift_results["drift_summary"]["high_drift_features"].append(f"numerical_{feature}")
            elif data["drift_severity"] == "medium":
                drift_results["drift_summary"]["medium_drift_features"].append(f"numerical_{feature}")
            else:
                drift_results["drift_summary"]["low_drift_features"].append(f"numerical_{feature}")
    
    for feature, data in drift_results["categorical_drift"].items():
        if isinstance(data, dict) and "drift_severity" in data:
            if data["drift_severity"] == "high":
                drift_results["drift_summary"]["high_drift_features"].append(f"categorical_{feature}")
            elif data["drift_severity"] == "medium":
                drift_results["drift_summary"]["medium_drift_features"].append(f"categorical_{feature}")
            else:
                drift_results["drift_summary"]["low_drift_features"].append(f"categorical_{feature}")
    
    log.info(f"Overall drift score: {drift_results['overall_drift_score']:.2%}")
    log.info(f"Drifted features: {drifted_features}/{total_features}")
    
    return drift_results

def calculate_distribution_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    features: List[str]
) -> Dict[str, Any]:
    """
    Calculate distribution drift using population stability index (PSI) and other metrics.
    
    Args:
        reference_data: Reference dataset
        current_data: Current dataset
        features: List of feature names to analyze
        
    Returns:
        Dictionary containing distribution drift metrics
    """
    
    log.info("Calculating distribution drift metrics")
    
    distribution_results = {
        "psi_scores": {},
        "wasserstein_distances": {},
        "distribution_summary": {}
    }
    
    for feature in features:
        if feature in reference_data.columns and feature in current_data.columns:
            try:
                ref_values = reference_data[feature].dropna()
                cur_values = current_data[feature].dropna()
                
                if len(ref_values) > 0 and len(cur_values) > 0:
                    # Calculate PSI (Population Stability Index)
                    psi_score = calculate_psi(ref_values, cur_values)
                    
                    # Calculate Wasserstein distance
                    wasserstein_dist = stats.wasserstein_distance(ref_values, cur_values)
                    
                    distribution_results["psi_scores"][feature] = {
                        "psi": psi_score,
                        "interpretation": interpret_psi(psi_score)
                    }
                    
                    distribution_results["wasserstein_distances"][feature] = wasserstein_dist
                    
                    log.info(f"Distribution drift for {feature}: PSI={psi_score:.4f}, Wasserstein={wasserstein_dist:.4f}")
                    
            except Exception as e:
                log.warning(f"Error calculating distribution drift for {feature}: {e}")
                distribution_results["psi_scores"][feature] = {"error": str(e)}
    
    return distribution_results

def calculate_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI).
    
    Args:
        reference: Reference data series
        current: Current data series
        bins: Number of bins for histogram
        
    Returns:
        PSI score
    """
    try:
        # Create bins based on reference data
        bin_edges = np.histogram_bin_edges(reference, bins=bins)
        
        # Calculate histograms
        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        cur_hist, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to probabilities
        ref_probs = ref_hist / ref_hist.sum()
        cur_probs = cur_hist / cur_hist.sum()
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        ref_probs = ref_probs + epsilon
        cur_probs = cur_probs + epsilon
        
        # Calculate PSI
        psi = np.sum((cur_probs - ref_probs) * np.log(cur_probs / ref_probs))
        
        return psi
        
    except Exception as e:
        log.warning(f"Error calculating PSI: {e}")
        return np.nan

def interpret_psi(psi: float) -> str:
    """
    Interpret PSI score.
    
    Args:
        psi: PSI score
        
    Returns:
        Interpretation string
    """
    if psi < 0.1:
        return "No significant population change"
    elif psi < 0.25:
        return "Slight population change"
    elif psi < 0.5:
        return "Moderate population change"
    else:
        return "Significant population change"

def generate_drift_report(
    drift_results: Dict[str, Any],
    distribution_results: Dict[str, Any],
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Generate a comprehensive drift report.
    
    Args:
        drift_results: Statistical drift results
        distribution_results: Distribution drift results
        reference_data: Reference dataset
        current_data: Current dataset
        
    Returns:
        Comprehensive drift report
    """
    
    log.info("Generating comprehensive drift report")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "reference_data_shape": reference_data.shape,
        "current_data_shape": current_data.shape,
        "statistical_drift": drift_results,
        "distribution_drift": distribution_results,
        "recommendations": [],
        "alerts": []
    }
    
    # Generate recommendations based on drift severity
    overall_drift_score = drift_results.get("overall_drift_score", 0.0)
    
    if overall_drift_score > 0.5:
        report["alerts"].append("HIGH_DRIFT_ALERT: More than 50% of features show significant drift")
        report["recommendations"].append("Consider retraining the model with updated data")
        report["recommendations"].append("Investigate data quality issues in drifted features")
    elif overall_drift_score > 0.25:
        report["alerts"].append("MEDIUM_DRIFT_ALERT: Significant drift detected in multiple features")
        report["recommendations"].append("Monitor model performance closely")
        report["recommendations"].append("Consider feature engineering adjustments")
    elif overall_drift_score > 0.1:
        report["alerts"].append("LOW_DRIFT_ALERT: Minor drift detected")
        report["recommendations"].append("Continue monitoring for trend changes")
    else:
        report["recommendations"].append("No significant drift detected - model should perform well")
    
    # Add specific recommendations for high-drift features
    high_drift_features = drift_results.get("drift_summary", {}).get("high_drift_features", [])
    if high_drift_features:
        report["recommendations"].append(f"High-drift features requiring attention: {', '.join(high_drift_features)}")
    
    # Add PSI-based recommendations
    high_psi_features = []
    for feature, data in distribution_results.get("psi_scores", {}).items():
        if isinstance(data, dict) and "psi" in data:
            if data["psi"] > 0.5:
                high_psi_features.append(feature)
    
    if high_psi_features:
        report["recommendations"].append(f"Features with significant distribution changes: {', '.join(high_psi_features)}")
    
    log.info(f"Generated drift report with {len(report['recommendations'])} recommendations")
    
    return report

def detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str]
) -> Dict[str, Any]:
    """
    Main function to detect data drift between reference and current data.
    
    Args:
        reference_data: Reference dataset (training data)
        current_data: Current dataset (new data)
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        
    Returns:
        Comprehensive drift analysis results
    """
    
    log.info("Starting comprehensive data drift detection")
    log.info(f"Reference data shape: {reference_data.shape}")
    log.info(f"Current data shape: {current_data.shape}")
    
    # Calculate statistical drift
    drift_results = calculate_statistical_drift(
        reference_data, current_data, numerical_features, categorical_features
    )
    
    # Calculate distribution drift
    all_features = numerical_features + categorical_features
    distribution_results = calculate_distribution_drift(
        reference_data, current_data, all_features
    )
    
    # Generate comprehensive report
    drift_report = generate_drift_report(
        drift_results, distribution_results, reference_data, current_data
    )
    
    # Combine all results
    final_results = {
        "drift_analysis": drift_results,
        "distribution_analysis": distribution_results,
        "drift_report": drift_report,
        "analysis_timestamp": datetime.now().isoformat()
    }
    
    log.info("Data drift detection completed successfully")
    
    return final_results 