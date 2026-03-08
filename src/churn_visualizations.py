import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import get_feature_types, save_plot

def analyze_numerical_features(data, numerical_features):
    """Bar chart of top 10 numeric features by |correlation| with Churn. Returns correlation series."""
    churn_correlation = data[numerical_features + ['Churn']].corr()['Churn'].sort_values(key=abs, ascending=False)
    top_correlations = churn_correlation.abs().head(10).sort_values(ascending=True)
    top_correlations = churn_correlation[top_correlations.index]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#c44f49' if x < 0 else '#75d18e' for x in top_correlations.values]
    ax.barh(range(len(top_correlations)), top_correlations.values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_correlations)))
    ax.set_yticklabels(top_correlations.index)
    ax.set_xlabel('Correlation with Churn', fontsize=12)
    ax.set_title('Top Features Correlated with Churn', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, 'churn_correlation.png')
    plt.close()
    
    return churn_correlation

def analyze_categorical_features(data, categorical_features):
    """Churn rate by category (up to 3 features)."""
    churn_rates_by_category = {}
    top_features = categorical_features[:3] if len(categorical_features) > 3 else categorical_features
    
    for feature in top_features:
        churn_by_category = data.groupby(feature)['Churn'].mean() * 100
        churn_rates_by_category[feature] = churn_by_category.sort_values(ascending=False)
    
    if churn_rates_by_category:
        fig, axes = plt.subplots(1, len(top_features), figsize=(6*len(top_features), 6))
        if len(top_features) == 1:
            axes = [axes]
        
        for i, feature in enumerate(top_features):
            churn_rates = churn_rates_by_category[feature]
            bars = axes[i].bar(range(len(churn_rates)), churn_rates.values, 
                            color='#49c4aa', alpha=0.7, edgecolor='#2a8a6f')
            axes[i].set_xticks(range(len(churn_rates)))
            axes[i].set_xticklabels(churn_rates.index, rotation=45, ha='right')
            axes[i].set_ylabel('Churn Rate (%)', fontsize=11)
            axes[i].set_title(f'Churn Rate by {feature}', fontsize=12, fontweight='bold')
            axes[i].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_plot(fig, 'churn_rate_by_category.png')
        plt.close()
    
    return churn_rates_by_category
