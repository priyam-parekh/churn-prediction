import pandas as pd
import numpy as np
import warnings
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from data_preprocessing import ChurnDataPreprocessor
from gbdt_models import GBDTModels
from model_evaluation import ModelEvaluator
from artifacts import save_serving_bundle
from feature_importance import FeatureImportanceAnalyzer
from model_validation import ModelValidator
from utils import create_results_dir, get_feature_types, plots_enabled, set_save_plots
from churn_visualizations import analyze_numerical_features, analyze_categorical_features

warnings.filterwarnings('ignore')

class ChurnPredictionPipeline:
    """Load data, EDA, preprocess, train XGB/LGB, evaluate, plot importance."""

    def __init__(self, data_path="../customer-churn-dataset/", results_path="../results/",
                 sample_size=None, save_artifacts=True, artifacts_subdir="serving"):
        # paths resolved relative to this file so it works from repo root or src/
        src_dir = Path(__file__).parent
        if isinstance(data_path, str):
            self.data_path = str((src_dir / data_path).resolve())
        else:
            self.data_path = str(Path(data_path).resolve())
        
        if isinstance(results_path, str):
            self.results_path = str((src_dir / results_path).resolve())
        else:
            self.results_path = str(Path(results_path).resolve())
        self.sample_size = sample_size
        self.processed_data = None
        self.models = {}
        self.evaluation_results = {}
        self.churn_correlation = None
        self.save_artifacts = save_artifacts
        self.artifacts_subdir = artifacts_subdir

        create_results_dir()
        
    def sample_data(self, data, sample_size=None):
        """Subsample while keeping churn ratio roughly intact."""
        if sample_size is None:
            sample_size = self.sample_size
            
        if sample_size and len(data) > sample_size:
            if 'Churn' in data.columns and data['Churn'].notna().any():
                churn_counts = data['Churn'].value_counts()
                if len(churn_counts) >= 2:
                    samples_per_class = min(sample_size // 2, churn_counts.min())
                    data = data.groupby('Churn', group_keys=False).apply(
                        lambda x: x.sample(min(len(x), samples_per_class), random_state=42)
                    )
                else:
                    data = data.sample(n=min(len(data), sample_size), random_state=42)
            else:
                data = data.sample(n=min(len(data), sample_size), random_state=42)
        return data
        
    def run_complete_analysis(self, optimize_hyperparameters=False, n_trials=20, optimize_for_accuracy=False):
        try:
            preprocessor = ChurnDataPreprocessor()
            combined_data = preprocessor.load_and_prepare_data(self.data_path)
            
            train_data = combined_data[combined_data['is_train'] == True].copy()
            test_data = combined_data[combined_data['is_train'] == False].copy()
            train_data = train_data.drop('is_train', axis=1)
            test_data = test_data.drop('is_train', axis=1)
            
            if self.sample_size is not None:
                print(f"Using subset: {self.sample_size} training samples, {self.sample_size // 4} test samples")
                train_data = self.sample_data(train_data)
                test_data = self.sample_data(test_data, sample_size=self.sample_size // 4)
            else:
                print(f"Using full dataset: {len(train_data)} training samples, {len(test_data)} test samples")
            
            feature_types = get_feature_types(train_data)
            churn_correlation = None
            if feature_types['numerical']:
                churn_correlation = analyze_numerical_features(train_data, feature_types['numerical'])
            if feature_types['categorical']:
                analyze_categorical_features(train_data, feature_types['categorical'])
            
            self.churn_correlation = churn_correlation
            
            train_data['is_train'] = True
            test_data['is_train'] = False
            combined_data = pd.concat([train_data, test_data], ignore_index=True)
            
            preprocessor.analyze_features(combined_data)
            combined_data = preprocessor.handle_missing_values(combined_data, fit_imputers=True)
            combined_data = preprocessor.encode_categorical_features(combined_data, fit_encoders=True)
            combined_data = preprocessor.create_feature_engineering(combined_data, fit_stats=True)
            
            feature_columns = preprocessor.select_features(combined_data)
            X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(combined_data, feature_columns)
            X_train, X_val, X_test = preprocessor.scale_features(X_train, X_val, X_test, 'standard')
            
            self.processed_data = {
                'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
                'feature_names': preprocessor.feature_names,
                'feature_types': preprocessor.feature_types
            }
            
            gbdt_trainer = GBDTModels(n_trials=n_trials, optimize_for_accuracy=optimize_for_accuracy)
            gbdt_trainer.train_all_models(X_train, y_train, X_val, y_val, optimize=optimize_hyperparameters)
            self.models.update(gbdt_trainer.models)
            
            validator = ModelValidator(cv_folds=5)
            validator.validate_all_models(self.models, X_train, y_train, X_val, y_val)
            
            evaluator = ModelEvaluator()
            evaluator.evaluate_all_models(self.models, X_train, y_train, X_val, y_val, 
                                         X_test, y_test, preprocessor.feature_names)
            
            primary_metric = 'accuracy' if optimize_for_accuracy else 'auc_roc'
            best_model_name, best_model_score = evaluator.get_best_model(primary_metric, 'test')
            self.evaluation_results = evaluator.evaluation_results
            
            evaluator.plot_roc_curves(self.models, X_train, y_train, X_val, y_val)
            best_model = self.models[best_model_name]
            evaluator.plot_confusion_matrix(best_model, X_val, y_val, best_model_name)
            
            importance_analyzer = FeatureImportanceAnalyzer()
            best_model = self.models[best_model_name]
            feature_names = preprocessor.feature_names
            top_churn_drivers = None
            feature_importance = None
            
            if hasattr(best_model, 'feature_importances_'):
                importance_scores = best_model.feature_importances_
            elif hasattr(best_model, 'coef_'):
                importance_scores = np.abs(best_model.coef_[0])
            else:
                importance_scores = None
            
            if importance_scores is not None:
                feature_importance = dict(zip(feature_names, importance_scores))
                top_churn_drivers = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
                # compare model importance to correlation
                if self.churn_correlation is not None:
                    self.compare_with_correlation_analysis(feature_importance, self.churn_correlation)
                
                self.make_research_graph(feature_importance, feature_names)
                
                if hasattr(best_model, 'feature_importances_'):
                    importance_analyzer.analyze_model_feature_importance(
                        best_model, X_train, y_train, X_val, y_val, best_model_name, feature_names
                    )
            
            self.print_summary(best_model_name, best_model_score, top_churn_drivers, optimize_for_accuracy=optimize_for_accuracy)
            self.save_results(
                preprocessor,
                gbdt_trainer,
                evaluator,
                importance_analyzer,
                best_model_name,
                X_train,
                y_train,
                optimize_for_accuracy=optimize_for_accuracy,
            )
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def print_summary(self, best_model_name, best_model_score, top_churn_drivers, optimize_for_accuracy=False):
        if optimize_for_accuracy:
            print(f"\nBest Model: {best_model_name} (Accuracy: {best_model_score:.2%})")
        else:
            print(f"\nBest Model: {best_model_name} (AUC: {best_model_score:.4f})")
        if top_churn_drivers:
            print("\nTop 10 Churn Drivers:")
            for i, (feature, importance) in enumerate(top_churn_drivers.items(), 1):
                print(f"  {i:2d}. {feature:<30} {importance:.4f}")
        print("\nModel Performance:")
        for model_name, results in self.evaluation_results.items():
            val_auc = results['val']['auc_roc']
            val_f1 = results['val']['f1']
            val_acc = results['val']['accuracy']
            if val_auc is not None and val_f1 is not None:
                acc_str = f" | Acc: {val_acc:.2%}" if val_acc is not None else ""
                print(f"  {model_name:<20} AUC: {val_auc:.4f} | F1: {val_f1:.4f}{acc_str}")
    
    def save_results(
        self,
        preprocessor,
        gbdt_trainer,
        evaluator,
        importance_analyzer,
        best_model_name,
        X_train,
        y_train,
        optimize_for_accuracy=False,
    ):
        if not self.save_artifacts:
            return
        primary_metric = "accuracy" if optimize_for_accuracy else "auc_roc"
        er = evaluator.evaluation_results.get(best_model_name, {})
        split_name = "test"
        pv = er.get("test", {}).get(primary_metric)
        if pv is None:
            split_name = "val"
            pv = er.get("val", {}).get(primary_metric)

        out_dir = Path(self.results_path) / self.artifacts_subdir
        path = save_serving_bundle(
            preprocessor,
            self.models[best_model_name],
            best_model_name,
            X_train,
            y_train,
            out_dir,
            primary_metric=primary_metric,
            primary_metric_value=float(pv) if pv is not None else None,
            split_for_metric=split_name,
        )
        print(f"\nServing artifacts written to: {path}")
    
    def make_research_graph(self, feature_importance, feature_names):
        """Plot top 10 churn predictors vs top 10 retention indicators side by side."""
        import matplotlib.pyplot as plt
        from utils import save_plot
        
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_churn_predictors = sorted_features[:10]
        top_retention_indicators = sorted_features[-10:]
        top_retention_indicators.reverse()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        churn_features = [f[0] for f in top_churn_predictors]
        churn_importance = [f[1] for f in top_churn_predictors]
        
        ax1.barh(range(len(churn_features)), churn_importance, color='#c44f49', alpha=0.8, edgecolor='#8a2e2a')
        ax1.set_yticks(range(len(churn_features)))
        ax1.set_yticklabels(churn_features, fontsize=10)
        ax1.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax1.set_title('Top 10 Churn Predictors', fontsize=13, fontweight='bold', pad=15)
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        retention_features = [f[0] for f in top_retention_indicators]
        retention_importance = [f[1] for f in top_retention_indicators]
        
        ax2.barh(range(len(retention_features)), retention_importance, color='#75d18e', alpha=0.8, edgecolor='#4a8a5f')
        ax2.set_yticks(range(len(retention_features)))
        ax2.set_yticklabels(retention_features, fontsize=10)
        ax2.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax2.set_title('Top 10 Retention Indicators', fontsize=13, fontweight='bold', pad=15)
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        plt.suptitle('Churn Predictors vs Retention Indicators', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_plot(fig, 'churn_predictors_vs_retention_indicators.png')
        plt.close()
    
    def compare_with_correlation_analysis(self, feature_importance, churn_correlation):
        """Compare model top features to correlation ranking."""
        top_model_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        if 'Churn' in churn_correlation.index:
            corr_without_target = churn_correlation.drop('Churn')
        else:
            corr_without_target = churn_correlation
        top_corr_features = corr_without_target.abs().sort_values(ascending=False).head(10)
        
        top_10_model = [f[0] for f in top_model_features]
        top_10_corr = top_corr_features.index.tolist()
        overlap = set(top_10_model) & set(top_10_corr)
        overlap_count = len(overlap)
        
        if overlap_count >= 5:
            print(f"  {overlap_count}/10 top features match correlation analysis")
        elif overlap_count >= 3:
            print(f"  {overlap_count}/10 top features match (model captures interactions)")
        else:
            print(f"  {overlap_count}/10 top features match (non-linear patterns detected)")
    
def main():
    parser = argparse.ArgumentParser(description="Churn prediction pipeline")
    parser.add_argument("--use-full-dataset", action="store_true", help="Use full dataset (~244K training samples)")
    parser.add_argument("--sample-size", type=int, default=None, help="Custom sample size (overrides --use-full-dataset if set)")
    parser.add_argument("--optimize-hyperparameters", action="store_true", help="Enable Optuna hyperparameter tuning")
    parser.add_argument("--n-trials", type=int, default=5, help="Number of Optuna trials (if optimization enabled)")
    parser.add_argument("--optimize-for-accuracy", action="store_true", help="Tune for accuracy (class weights + threshold); default is AUC")
    parser.add_argument("--no-save-artifacts", action="store_true", help="Skip writing serving bundle (model, preprocessor, SHAP background)")
    parser.add_argument("--artifacts-subdir", type=str, default="serving", help="Subdirectory under results/ for serving artifacts")
    args = parser.parse_args()

    if args.sample_size is not None:
        sample_size = args.sample_size
    elif args.use_full_dataset:
        sample_size = None  # None means use full dataset
    else:
        sample_size = 10000

    set_save_plots(True)

    pipeline = ChurnPredictionPipeline(
        sample_size=sample_size,
        save_artifacts=not args.no_save_artifacts,
        artifacts_subdir=args.artifacts_subdir,
    )
    pipeline.run_complete_analysis(
        optimize_hyperparameters=args.optimize_hyperparameters,
        n_trials=args.n_trials,
        optimize_for_accuracy=args.optimize_for_accuracy,
    )

if __name__ == "__main__":
    main()
