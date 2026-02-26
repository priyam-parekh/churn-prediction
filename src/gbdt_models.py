import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                           recall_score, f1_score)
import optuna
import warnings
warnings.filterwarnings('ignore')

def _scale_pos_weight(y_train):
    """Weight for positive class so the model doesn't ignore the minority class."""
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def _tune_threshold_for_accuracy(model, X_val, y_val):
    """Find probability threshold that maximizes validation accuracy (instead of default 0.5)."""
    proba = model.predict_proba(X_val)[:, 1]
    best_acc = 0.0
    best_thresh = 0.5
    for t in np.linspace(0.15, 0.85, 71):
        pred = (proba >= t).astype(int)
        acc = accuracy_score(y_val, pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    return best_thresh, best_acc


class GBDTModels:
    """Trains XGBoost and LightGBM with sensible defaults (subsample, colsample, max_depth). Optional Optuna tuning."""

    def __init__(self, random_state=42, n_trials=100, optimize_for_accuracy=False):
        self.random_state = random_state
        self.n_trials = n_trials
        self.optimize_for_accuracy = optimize_for_accuracy
        self.models = {}
        self.results = {}
        self.best_params = {}

    def _scoring(self):
        return 'accuracy' if self.optimize_for_accuracy else 'roc_auc'

    def optimize_xgboost(self, X_train, y_train, X_val, y_val):
        scale_pos = _scale_pos_weight(y_train)
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': self.random_state,
                'n_jobs': 1,
                'tree_method': 'hist',
                'scale_pos_weight': scale_pos,
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
            model = xgb.XGBClassifier(**params)
            cross_val_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=self._scoring())
            return cross_val_scores.mean()

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': self.random_state,
            'n_jobs': 1,
            'tree_method': 'hist',
            'scale_pos_weight': scale_pos,
        })
        self.best_params['XGBoost'] = best_params
        return best_params

    def optimize_lightgbm(self, X_train, y_train, X_val, y_val):
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'random_state': self.random_state,
                'n_jobs': 1,
                'verbosity': -1,
                'class_weight': 'balanced',
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
            model = lgb.LGBMClassifier(**params)
            cross_val_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=self._scoring())
            return cross_val_scores.mean()

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_params.update({
            'objective': 'binary',
            'metric': 'auc',
            'random_state': self.random_state,
            'n_jobs': 1,
            'verbosity': -1,
            'class_weight': 'balanced',
        })
        self.best_params['LightGBM'] = best_params
        return best_params

    def train_xgboost(self, X_train, y_train, X_val, y_val, use_optimized=False):
        scale_pos = _scale_pos_weight(y_train)
        if use_optimized and 'XGBoost' in self.best_params:
            params = self.best_params['XGBoost'].copy()
        else:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': self.random_state,
                'n_jobs': 1,
                'tree_method': 'hist',
                'scale_pos_weight': scale_pos,
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 500,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        thresh, _ = _tune_threshold_for_accuracy(model, X_val, y_val)
        model.best_threshold_ = thresh
        results = self.evaluate_model(model, X_train, y_train, X_val, y_val, 'XGBoost')
        self.results['XGBoost'] = results
        self.models['XGBoost'] = model
        return model

    def train_lightgbm(self, X_train, y_train, X_val, y_val, use_optimized=False):
        if use_optimized and 'LightGBM' in self.best_params:
            params = self.best_params['LightGBM'].copy()
        else:
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'random_state': self.random_state,
                'n_jobs': 1,
                'verbosity': -1,
                'class_weight': 'balanced',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 500,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        thresh, _ = _tune_threshold_for_accuracy(model, X_val, y_val)
        model.best_threshold_ = thresh
        results = self.evaluate_model(model, X_train, y_train, X_val, y_val, 'LightGBM')
        self.results['LightGBM'] = results
        self.models['LightGBM'] = model
        return model
    
    def evaluate_model(self, model, X_train, y_train, X_val, y_val, model_name):
        thresh = getattr(model, 'best_threshold_', 0.5)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_train_pred = (y_train_proba >= thresh).astype(int)
        y_val_pred = (y_val_proba >= thresh).astype(int)
        return {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'train_precision': precision_score(y_train, y_train_pred),
            'val_precision': precision_score(y_val, y_val_pred),
            'train_recall': recall_score(y_train, y_train_pred),
            'val_recall': recall_score(y_val, y_val_pred),
            'train_f1': f1_score(y_train, y_train_pred),
            'val_f1': f1_score(y_val, y_val_pred),
            'train_auc': roc_auc_score(y_train, y_train_proba),
            'val_auc': roc_auc_score(y_val, y_val_proba),
            'y_val_pred': y_val_pred,
            'y_val_proba': y_val_proba
        }
    
    def get_best_model(self):
        if not self.results:
            return None, None, None
        
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['val_auc'])
        best_model = self.models[best_model_name]
        best_score = self.results[best_model_name]['val_auc']
        
        return best_model_name, best_model, best_score
    
    def save_models(self, save_dir="../results/"):
        pass

    def train_all_models(self, X_train, y_train, X_val, y_val, optimize=True):
        if optimize:
            self.optimize_xgboost(X_train, y_train, X_val, y_val)
            self.optimize_lightgbm(X_train, y_train, X_val, y_val)
        
        self.train_xgboost(X_train, y_train, X_val, y_val, use_optimized=optimize)
        self.train_lightgbm(X_train, y_train, X_val, y_val, use_optimized=optimize)
