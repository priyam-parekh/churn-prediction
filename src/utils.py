import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

_SAVE_PLOTS = True
_SAVE_ARTIFACTS = False  # no CSV/JSON/TXT writes—keeps results/ to just plots

def set_save_plots(enabled: bool) -> None:
    global _SAVE_PLOTS
    _SAVE_PLOTS = bool(enabled)

def plots_enabled() -> bool:
    return _SAVE_PLOTS

def artifacts_enabled() -> bool:
    return False

def load_data(data_path="../customer-churn-dataset/"):
    data_path = Path(data_path)
    if not data_path.is_absolute():
        src_dir = Path(__file__).parent
        data_path = (src_dir / data_path).resolve()
    else:
        data_path = data_path.resolve()

    train_df = pd.read_csv(data_path / "train.csv")
    test_df = pd.read_csv(data_path / "test.csv")
    feature_descriptions = pd.read_csv(data_path / "data_descriptions.csv")
    return train_df, test_df, feature_descriptions

def create_results_dir(results_dir: str = "../results"):
    if not (plots_enabled() or artifacts_enabled()):
        return None
    src_dir = Path(__file__).parent
    results_path = (src_dir / results_dir).resolve()
    results_path.mkdir(parents=True, exist_ok=True)
    return results_path

def save_plot(fig, filename, dpi=300):
    if not plots_enabled():
        return
    results_dir = create_results_dir()
    if results_dir is None:
        return
    filepath = results_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')

def print_data_info(df, name="Dataset"):
    print(f"\n{'='*50}")
    print(f"{name} Information")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found")

def get_feature_types(df):
    """Return dict of numerical, categorical, binary, target. Skips CustomerID."""
    feature_types = {
        'numerical': [],
        'categorical': [],
        'binary': [],
        'target': []
    }
    
    for col in df.columns:
        if col == 'Churn':
            feature_types['target'].append(col)
        elif col == 'CustomerID':
            continue  # skip ID column
        elif df[col].dtype in ['int64', 'float64']:
            if df[col].nunique() == 2:
                feature_types['binary'].append(col)
            else:
                feature_types['numerical'].append(col)
        else:
            feature_types['categorical'].append(col)
    
    return feature_types

def plot_correlation_heatmap(df, features, title="Feature Correlation Heatmap"):
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[features].corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # upper triangle only
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return plt.gcf()

def plot_distribution(df, feature, target='Churn', plot_type='hist'):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    if plot_type == 'hist':
        for i, churn_value in enumerate([0, 1]):
            data = df[df[target] == churn_value][feature]
            axes[i].hist(data, bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{feature} Distribution - Churn: {churn_value}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
    
    elif plot_type == 'box':
        sns.boxplot(data=df, x=target, y=feature, ax=axes[0])
        axes[0].set_title(f'{feature} by Churn Status')
        sns.violinplot(data=df, x=target, y=feature, ax=axes[1])
        axes[1].set_title(f'{feature} Distribution by Churn Status')
    
    plt.tight_layout()
    return fig

