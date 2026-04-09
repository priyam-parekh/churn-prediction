"""
FastAPI churn prediction with optional per-row SHAP (TreeExplainer + saved background).
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from artifacts import load_serving_bundle


def _artifacts_dir() -> str:
    return os.environ.get("CHURN_ARTIFACTS_DIR", "results/serving")


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = _artifacts_dir()
    try:
        bundle = load_serving_bundle(path)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Could not load artifacts from {path}. Train the model first (python src/main.py) "
            "or set CHURN_ARTIFACTS_DIR to a directory containing the serving bundle."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to load serving bundle from {path}: {e}") from e

    model = bundle["model"]
    bg = bundle["shap_background"]
    # "interventional" works with a modest background sample; "tree_path_dependent"
    # can error if the background does not cover all tree leaves.
    explainer = shap.TreeExplainer(
        model,
        bg,
        feature_perturbation="interventional",
    )

    app.state.bundle = bundle
    app.state.explainer = explainer
    yield


app = FastAPI(title="Churn prediction API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Raw feature values (pre-engineering), same names as training CSV.")
    include_shap: bool = True


class ShapContribution(BaseModel):
    feature: str
    value: float
    shap: float


class PredictResponse(BaseModel):
    churn_probability: float
    churn_binary: int
    model_name: str
    best_threshold: float
    expected_value: float | None = None
    shap_contributions: list[ShapContribution] | None = None


def _validate_features(bundle: dict, features: dict[str, Any]) -> None:
    required = set(bundle["predict_input"].get("required_columns", []))
    if not required:
        required = set(bundle["preprocessor"].raw_input_column_names())
    missing = required - set(features.keys())
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required feature keys: {sorted(missing)}",
        )


def _compute_shap(explainer, X_np: np.ndarray, feature_names: list[str]) -> tuple[float, list[dict[str, Any]]]:
    sv = explainer.shap_values(X_np)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)

    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = np.asarray(ev).ravel()
        expected_value = float(ev[1] if len(ev) > 1 else ev[0])
    else:
        expected_value = float(ev)

    row = sv[0]
    contributions = [
        {
            "feature": feature_names[i],
            "value": float(X_np[0, i]),
            "shap": float(row[i]),
        }
        for i in range(len(feature_names))
    ]
    contributions.sort(key=lambda x: abs(x["shap"]), reverse=True)
    return expected_value, contributions


def _predict_sync(body: PredictRequest, bundle, explainer) -> PredictResponse:
    _validate_features(bundle, body.features)
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    meta = bundle["metadata"]
    threshold = float(meta.get("best_threshold", getattr(model, "best_threshold_", 0.5)))

    X = preprocessor.transform_raw(body.features)
    X_np = np.asarray(X, dtype=np.float64)
    proba = float(model.predict_proba(X_np)[0, 1])
    binary = int(proba >= threshold)
    feat_names = list(X.columns)

    expected_value = None
    contributions: list[ShapContribution] | None = None

    if body.include_shap:
        expected_value, raw_contrib = _compute_shap(explainer, X_np, feat_names)
        contributions = [ShapContribution(**c) for c in raw_contrib]

    return PredictResponse(
        churn_probability=proba,
        churn_binary=binary,
        model_name=meta.get("best_model_name", "unknown"),
        best_threshold=threshold,
        expected_value=expected_value,
        shap_contributions=contributions,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schema")
def schema():
    bundle = app.state.bundle
    prep = bundle["predict_input"]
    return {
        "required_columns": prep.get("required_columns", bundle["preprocessor"].raw_input_column_names()),
        "model_name": bundle["metadata"].get("best_model_name"),
        "feature_order": bundle["preprocessor"].feature_names,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest):
    bundle = app.state.bundle
    explainer = app.state.explainer

    try:
        return await asyncio.to_thread(_predict_sync, body, bundle, explainer)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
