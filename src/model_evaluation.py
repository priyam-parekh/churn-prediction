import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                           recall_score, f1_score, roc_curve, confusion_matrix)
from utils import save_plot

class ModelEvaluator:
    """Compute AUC, F1, precision, recall; pick best model. AUC primary for imbalance."""

    def __init__(self):
        self.evaluation_results = {}
        self.predictions = {}

    def evaluate_model(self, model, X_train, y_train, X_val, y_val, X_test, y_test, model_name):
        thresh = getattr(model, 'best_threshold_', 0.5)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        y_train_pred = (y_train_proba >= thresh).astype(int)
        y_val_pred = (y_val_proba >= thresh).astype(int)
        y_test_pred = (y_test_proba >= thresh).astype(int)
        
        results = {}
        for split_name, y, y_pred, y_proba in [
            ('train', y_train, y_train_pred, y_train_proba),
            ('val', y_val, y_val_pred, y_val_proba),
            ('test', y_test, y_test_pred, y_test_proba)
        ]:
            # test can be empty or unlabeled (e.g. Kaggle test set)
            if y.isna().any() or len(y) == 0:
                results[split_name] = {
                    'accuracy': None, 'precision': None, 'recall': None,
                    'f1': None, 'auc_roc': None
                }
            else:
                results[split_name] = {
                    'accuracy': accuracy_score(y, y_pred),
                    'precision': precision_score(y, y_pred),
                    'recall': recall_score(y, y_pred),
                    'f1': f1_score(y, y_pred),
                    'auc_roc': roc_auc_score(y, y_proba)
                }
        
        self.evaluation_results[model_name] = results
        self.predictions[model_name] = {
            'y_val': y_val,
            'y_val_proba': y_val_proba,
            'y_val_pred': y_val_pred,
            'y_train': y_train,
            'y_train_proba': y_train_proba
        }
        
        return results

    def evaluate_all_models(self, models, X_train, y_train, X_val, y_val, X_test, y_test, feature_names):
        for model_name, model in models.items():
            self.evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name)
    
    def get_best_model(self, metric='auc_roc', split='test'):
        """Best model by metric; falls back to val if test has no labels."""
        if not self.evaluation_results:
            return None, None
        
        if split == 'test':
            has_test_scores = any(
                self.evaluation_results[model_name][split][metric] is not None 
                for model_name in self.evaluation_results.keys()
            )
            if not has_test_scores:
                split = 'val'
        
        valid_model_scores = {
            model_name: results[split][metric] 
            for model_name, results in self.evaluation_results.items()
            if results[split][metric] is not None
        }
        
        if not valid_model_scores:
            return None, None
        
        best_model_name = max(valid_model_scores.keys(), key=lambda x: valid_model_scores[x])
        best_score = valid_model_scores[best_model_name]
        
        return best_model_name, best_score
    
    def plot_roc_curves(self, models, X_train, y_train, X_val, y_val):
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = {'XGBoost': '#4997c4', 'LightGBM': '#49c4aa'}

        for model_name, model in models.items():
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_val_proba = model.predict_proba(X_val)[:, 1]
            fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
            fpr_val, tpr_val, _ = roc_curve(y_val, y_val_proba)
            auc_train = roc_auc_score(y_train, y_train_proba)
            auc_val = roc_auc_score(y_val, y_val_proba)
            color = colors.get(model_name, '#4997c4')
            ax.plot(fpr_val, tpr_val, color=color, linewidth=2,
                    label=f'{model_name} (Val, AUC={auc_val:.3f})')
            ax.plot(fpr_train, tpr_train, color=color, linewidth=1,
                    linestyle='--', alpha=0.6, label=f'{model_name} (Train, AUC={auc_train:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        save_plot(fig, 'roc_curves.png')
        plt.close()
    
    def plot_confusion_matrix(self, best_model, X_val, y_val, best_model_name):
        import seaborn as sns
        thresh = getattr(best_model, 'best_threshold_', 0.5)
        y_proba = best_model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= thresh).astype(int)
        cm = confusion_matrix(y_val, y_pred)
        fig, ax = plt.subplots(figsize=(7, 6))
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        from matplotlib.colors import LinearSegmentedColormap
        colors_list = ['#f0f8ff', '#87ceeb', '#4997c4', '#2c5f7d']
        cmap = LinearSegmentedColormap.from_list('custom_blue', colors_list, N=256)
        sns.heatmap(cm_percent, annot=False, fmt='', cmap=cmap,
                    cbar_kws={'label': 'Percentage (%)', 'shrink': 0.8},
                    linewidths=3, linecolor='white',
                    square=True, ax=ax, vmin=0, vmax=100)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm_percent[i, j]
                count = int(cm[i, j])
                text_color = "white" if value > 35 else "#1a1a1a"
                ax.text(j + 0.5, i + 0.5, f'{count}\n({value:.1f}%)',
                       ha="center", va="center",
                       color=text_color,
                       fontsize=14, fontweight='bold',
                       family='sans-serif')
        
        ax.set_xticks([0.5, 1.5])
        ax.set_yticks([0.5, 1.5])
        ax.set_xticklabels(['No Churn', 'Churn'], fontsize=12, fontweight='bold')
        ax.set_yticklabels(['No Churn', 'Churn'], fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
        ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=15, fontweight='bold', pad=15)
        
        plt.tight_layout()
        save_plot(fig, 'confusion_matrix.png')
        plt.close()
