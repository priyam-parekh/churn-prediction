"""
Persist preprocessor, deployed model, SHAP background matrix, and metadata for serving.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from data_preprocessing import ChurnDataPreprocessor


def _resolve_bundle_path(artifacts_dir: str | Path) -> Path:
    """Relative paths are resolved from the project root (parent of ``src/``)."""
    path = Path(artifacts_dir)
    if path.is_absolute():
        return path.resolve()
    repo_root = Path(__file__).resolve().parent.parent
    return (repo_root / path).resolve()


def _package_versions() -> dict:
    out = {}
    try:
        import sklearn

        out["sklearn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import xgboost

        out["xgboost"] = xgboost.__version__
    except Exception:
        pass
    try:
        import lightgbm

        out["lightgbm"] = lightgbm.__version__
    except Exception:
        pass
    return out


def sample_shap_background(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n: int = 256,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Proportional stratified sample from training rows for TreeExplainer background.
    If len(X_train) <= n, returns all rows.
    """
    if len(X_train) <= n:
        return X_train.copy(), y_train.copy()

    yv = np.asarray(y_train).ravel()
    if len(np.unique(yv)) < 2:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X_train), size=n, replace=False)
        return X_train.iloc[idx].reset_index(drop=True), y_train.iloc[idx].reset_index(drop=True)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=random_state)
    train_idx, _ = next(sss.split(np.zeros(len(yv)), yv))
    return (
        X_train.iloc[train_idx].reset_index(drop=True),
        y_train.iloc[train_idx].reset_index(drop=True),
    )


def save_serving_bundle(
    preprocessor: ChurnDataPreprocessor,
    model,
    best_model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    out_dir: str | Path,
    *,
    background_size: int = 256,
    random_state: int = 42,
    primary_metric: str = "auc_roc",
    primary_metric_value: float | None = None,
    split_for_metric: str = "test",
) -> Path:
    """
    Write preprocessor pickles, single model joblib, shap_background.joblib, metadata.json.
    """
    out_path = _resolve_bundle_path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    preprocessor.save_preprocessing_artifacts(str(out_path))

    joblib.dump(model, out_path / "churn_model.joblib")

    X_bg, _ = sample_shap_background(
        X_train, y_train, n=background_size, random_state=random_state
    )
    joblib.dump(
        {
            "X": np.asarray(X_bg, dtype=np.float64),
            "feature_names": list(X_bg.columns),
        },
        out_path / "shap_background.joblib",
    )

    threshold = float(getattr(model, "best_threshold_", 0.5))
    meta = {
        "best_model_name": best_model_name,
        "primary_metric": primary_metric,
        "primary_metric_split": split_for_metric,
        "primary_metric_value": primary_metric_value,
        "best_threshold": threshold,
        "background_size": int(len(X_bg)),
        "background_requested": background_size,
        "versions": _package_versions(),
    }
    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    raw_cols = preprocessor.raw_input_column_names()
    with open(out_path / "predict_input_columns.json", "w", encoding="utf-8") as f:
        json.dump({"required_columns": raw_cols}, f, indent=2)

    return out_path


def load_serving_bundle(artifacts_dir: str | Path) -> dict:
    """Load everything needed to run inference + SHAP (next tasks)."""
    path = _resolve_bundle_path(artifacts_dir)

    preprocessor = ChurnDataPreprocessor()
    preprocessor.load_preprocessing_artifacts(str(path))

    model = joblib.load(path / "churn_model.joblib")
    bg = joblib.load(path / "shap_background.joblib")
    with open(path / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    pred_cols_path = path / "predict_input_columns.json"
    predict_input = {"required_columns": list(preprocessor.raw_input_column_names())}
    if pred_cols_path.exists():
        with open(pred_cols_path, encoding="utf-8") as f:
            predict_input = json.load(f)

    return {
        "preprocessor": preprocessor,
        "model": model,
        "shap_background": bg["X"],
        "shap_background_columns": bg.get("feature_names"),
        "metadata": metadata,
        "predict_input": predict_input,
        "artifacts_dir": str(path),
    }
