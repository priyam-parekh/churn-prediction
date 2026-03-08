import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import save_plot

class FeatureImportanceAnalyzer:
    """Extract feature_importances_ from GBDT and plot top N."""

    def __init__(self):
        self.feature_importance_results = {}

    def analyze_model_feature_importance(self, model, X_train, y_train, X_val, y_val,
                                         model_name, feature_names):
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance_scores = np.abs(model.coef_[0])
        else:
            return
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        self.plot_importance(importance_df, model_name)
        
        self.feature_importance_results[model_name] = {
            'importance_scores': importance_scores,
            'importance_df': importance_df
        }
    
    def plot_importance(self, importance_df, model_name, top_n=15):
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'], 
                color='#4997c4', edgecolor='#2d5f7a')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.title(f'Feature Importance - {model_name} (Top {top_n})')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        save_plot(plt.gcf(), 'feature_importance.png')
        plt.close()
    
    def save_analysis_results(self, save_dir="../results/"):
        pass
