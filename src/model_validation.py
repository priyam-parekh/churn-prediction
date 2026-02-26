import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')


class ModelValidator:
    """Stratified K-fold CV for stability; basic checks for imbalance, missing values, constant features, train/val drift."""

    def __init__(self, cv_folds=5, random_state=42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.validation_results = {}
    
    def cross_validate_model(self, model, X_train, y_train, model_name):
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=1)
        
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        self.validation_results[model_name] = {
            'cv_mean': mean_score,
            'cv_std': std_score,
            'cv_scores': cv_scores
        }
        return mean_score, std_score

    def validate_data_quality(self, X_train, y_train, X_val, y_val):
        issues = []
        
        churn_rate = y_train.mean()
        if churn_rate < 0.1 or churn_rate > 0.9:
            issues.append(f"Severe class imbalance: {churn_rate:.2%} churn rate")
        
        if X_train.isnull().sum().sum() > 0:
            issues.append("Missing values detected in training data")
        
        constant_features = X_train.columns[X_train.nunique() <= 1].tolist()
        if constant_features:
            issues.append(f"Constant features found: {constant_features}")
        
        if len(X_train.columns) > 0:
            train_mean = X_train.mean()
            val_mean = X_val.mean()
            large_diff = (train_mean - val_mean).abs() > (train_mean.abs() * 0.5)
            if large_diff.any():
                issues.append("Train/validation distribution mismatch detected")

        if issues:
            print("\nWarnings:")
            for issue in issues:
                print(f"  - {issue}")
        
        return issues
    
    def check_overfitting(self, train_score, val_score, threshold=0.1):
        gap = train_score - val_score
        if gap > threshold:
            return True, gap
        return False, gap
    
    def validate_all_models(self, models, X_train, y_train, X_val, y_val):
        issues = self.validate_data_quality(X_train, y_train, X_val, y_val)
        for model_name, model in models.items():
            self.cross_validate_model(model, X_train, y_train, model_name)
        
        return self.validation_results

