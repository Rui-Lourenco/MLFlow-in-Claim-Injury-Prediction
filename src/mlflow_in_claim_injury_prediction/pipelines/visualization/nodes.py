"""Visualization nodes for MLOps pipeline."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import logging
import os

logger = logging.getLogger(__name__)

def create_feature_importance_comparison(feature_importance_summary: pd.DataFrame):
    """Create side-by-side feature importance comparison."""
    
    logger.info("Creating feature importance comparison visualization")
    
    # Ensure output directory exists
    os.makedirs('data/08_reporting', exist_ok=True)
    
    # Get top 15 features by average importance
    top_features = feature_importance_summary.nlargest(15, 'avg_importance')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(top_features))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, top_features['xgb_importance'], width, 
                   label='XGBoost', alpha=0.8, color='#1f77b4')
    bars2 = ax.bar(x + width/2, top_features['rf_importance'], width,
                   label='Random Forest', alpha=0.8, color='#ff7f0e')
    
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Importance Score', fontsize=12)
    ax.set_title('Feature Importance Comparison: XGBoost vs Random Forest', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_features['feature'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/08_reporting/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("✅ Feature importance comparison saved to data/08_reporting/feature_importance_comparison.png")

def create_confusion_matrix_heatmap(confusion_matrix: np.ndarray):
    """Create professional confusion matrix heatmap."""
    
    logger.info("Creating confusion matrix heatmap")
    
    # Ensure output directory exists
    os.makedirs('data/08_reporting', exist_ok=True)
    
    # Define class labels (adjust based on your classes)
    class_labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 
                   'Class 5', 'Class 6', 'Class 7', 'Class 8']
    
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Normalized Frequency'})
    
    plt.title('Confusion Matrix - Workers Compensation Claim Classification', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.tight_layout()
    plt.savefig('data/08_reporting/confusion_matrix_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("✅ Confusion matrix heatmap saved to data/08_reporting/confusion_matrix_heatmap.png")

def create_model_performance_dashboard(test_metrics: dict, test_report: dict):
    """Create interactive model performance dashboard."""
    
    logger.info("Creating interactive model performance dashboard")
    
    # Ensure output directory exists
    os.makedirs('data/08_reporting', exist_ok=True)
    
    # Extract class-wise metrics
    classes = [k for k in test_report.keys() if k.isdigit()]
    class_metrics = []
    
    for cls in classes:
        class_metrics.append({
            'Class': f'Class {cls}',
            'Precision': test_report[cls]['precision'],
            'Recall': test_report[cls]['recall'],
            'F1-Score': test_report[cls]['f1-score'],
            'Support': test_report[cls]['support']
        })
    
    df_metrics = pd.DataFrame(class_metrics)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Class-wise Performance', 'Overall Metrics', 
                       'Support Distribution', 'Precision vs Recall'),
        specs=[[{"secondary_y": False}, {"type": "indicator"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Class-wise performance
    fig.add_trace(
        go.Bar(x=df_metrics['Class'], y=df_metrics['Precision'], 
               name='Precision', marker_color='lightblue'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=df_metrics['Class'], y=df_metrics['Recall'], 
               name='Recall', marker_color='lightgreen'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=df_metrics['Class'], y=df_metrics['F1-Score'], 
               name='F1-Score', marker_color='lightcoral'),
        row=1, col=1
    )
    
    # Overall metrics indicators
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=test_metrics['accuracy'],
            title={'text': f"Accuracy<br>{test_metrics['accuracy']:.1%}"},
            gauge={'axis': {'range': [None, 1]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                            {'range': [0.5, 0.8], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 0.9}}
        ),
        row=1, col=2
    )
    
    # Support distribution
    fig.add_trace(
        go.Bar(x=df_metrics['Class'], y=df_metrics['Support'],
               marker_color='purple', name='Support'),
        row=2, col=1
    )
    
    # Precision vs Recall scatter
    fig.add_trace(
        go.Scatter(x=df_metrics['Recall'], y=df_metrics['Precision'],
                   mode='markers+text', text=df_metrics['Class'],
                   textposition="top center", marker_size=10,
                   marker_color='red', name='Classes'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, 
                      title_text="Workers Compensation Model Performance Dashboard")
    
    fig.write_html('data/08_reporting/model_performance_dashboard.html')
    
    logger.info("✅ Interactive performance dashboard saved to data/08_reporting/model_performance_dashboard.html")

def create_data_drift_visualization(data_drift_report: dict):
    """Create data drift monitoring visualization."""
    
    logger.info("Creating data drift visualization")
    
    # Ensure output directory exists
    os.makedirs('data/08_reporting', exist_ok=True)
    
    drift_data = data_drift_report['drift_analysis']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Overall drift score
    ax1.bar(['Overall Drift Score'], [drift_data['overall_drift_score']], 
            color='green' if drift_data['overall_drift_score'] < 0.1 else 'red',
            alpha=0.7)
    ax1.set_title('Overall Data Drift Score', fontweight='bold')
    ax1.set_ylabel('Drift Score')
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.1, color='orange', linestyle='--', label='Warning Threshold')
    ax1.axhline(y=0.25, color='red', linestyle='--', label='Critical Threshold')
    ax1.legend()
    
    # Drift summary
    summary = drift_data['drift_summary']
    categories = ['Total Features', 'Drifted Features']
    values = [summary['total_features'], summary['drifted_features']]
    colors = ['lightblue', 'red']
    
    ax2.bar(categories, values, color=colors, alpha=0.7)
    ax2.set_title('Feature Drift Summary', fontweight='bold')
    ax2.set_ylabel('Count')
    
    # Drift percentage pie chart
    if summary['drift_percentage'] > 0:
        labels = ['No Drift', 'Drift Detected']
        sizes = [100 - summary['drift_percentage'], summary['drift_percentage']]
        colors = ['lightgreen', 'red']
    else:
        labels = ['No Drift Detected']
        sizes = [100]
        colors = ['lightgreen']
    
    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Drift Percentage Distribution', fontweight='bold')
    
    # Timestamp info
    timestamp = data_drift_report.get('analysis_timestamp', 'Unknown')
    ax4.text(0.1, 0.8, f"Analysis Timestamp:\n{timestamp}", fontsize=12, 
             transform=ax4.transAxes, verticalalignment='top')
    ax4.text(0.1, 0.6, f"Reference Data Shape:\n{data_drift_report['drift_report']['reference_data_shape']}", 
             fontsize=12, transform=ax4.transAxes, verticalalignment='top')
    ax4.text(0.1, 0.4, f"Current Data Shape:\n{data_drift_report['drift_report']['current_data_shape']}", 
             fontsize=12, transform=ax4.transAxes, verticalalignment='top')
    
    recommendations = data_drift_report['drift_report'].get('recommendations', [])
    if recommendations:
        ax4.text(0.1, 0.2, f"Recommendations:\n• {recommendations[0]}", 
                fontsize=12, transform=ax4.transAxes, verticalalignment='top')
    
    ax4.set_title('Analysis Summary', fontweight='bold')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('data/08_reporting/data_drift_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("✅ Data drift visualization saved to data/08_reporting/data_drift_analysis.png")

def create_permutation_importance_plot(permutation_importance: pd.DataFrame):
    """Create permutation importance visualization."""
    
    logger.info("Creating permutation importance plot")
    
    # Ensure output directory exists
    os.makedirs('data/08_reporting', exist_ok=True)
    
    # Sort by importance
    df_sorted = permutation_importance.sort_values('importance_mean', ascending=True)
    top_15 = df_sorted.tail(15)  # Get top 15
    
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar plot with error bars
    bars = plt.barh(range(len(top_15)), top_15['importance_mean'], 
                    xerr=top_15['importance_std'], capsize=5,
                    color='skyblue', alpha=0.8, edgecolor='navy')
    
    plt.yticks(range(len(top_15)), top_15['feature'])
    plt.xlabel('Permutation Importance Score', fontsize=12)
    plt.title('Top 15 Features - Permutation Importance Analysis', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars, top_15['importance_mean'], top_15['importance_std'])):
        plt.text(val + std + 0.001, i, f'{val:.3f}', 
                va='center', ha='left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('data/08_reporting/permutation_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("✅ Permutation importance plot saved to data/08_reporting/permutation_importance.png")