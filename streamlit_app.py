"""
Streamlit UI for churn prediction — calls the FastAPI service (no local model import).
Run API first: PYTHONPATH=src uvicorn api.app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_API = os.environ.get("FASTAPI_URL", "http://127.0.0.1:8000")

WIDGET_PREFIX = "feat_"


@st.cache_data(show_spinner=False)
def _load_training_frame() -> pd.DataFrame:
    train_csv = ROOT / "customer-churn-dataset" / "train.csv"
    if not train_csv.exists():
        return pd.DataFrame()
    return pd.read_csv(train_csv)


def _widget_key(col: str) -> str:
    return f"{WIDGET_PREFIX}{col}"


def _column_ui_kind(series: pd.Series) -> str:
    """numeric (slider), binary (selectbox), or categorical (selectbox)."""
    s = series.dropna()
    if len(s) == 0:
        return "categorical"
    dt = series.dtype
    if dt in ("int64", "int32", "float64", "float32"):
        if s.nunique() <= 2:
            return "binary"
        return "numeric"
    return "categorical"


def dt_is_int_like(s: pd.Series) -> bool:
    return s.dtype in ("int64", "int32", "int8", "bool")


def _numeric_is_discrete_integers(s: pd.Series) -> bool:
    if not dt_is_int_like(s):
        return False
    t = s.dropna()
    if len(t) == 0:
        return False
    return bool(t.apply(lambda x: float(x).is_integer()).all())


def _init_feature_defaults(required: list[str], train_df: pd.DataFrame) -> None:
    for col in required:
        key = _widget_key(col)
        if key in st.session_state:
            continue
        if train_df.empty or col not in train_df.columns:
            st.session_state[key] = 0.0
            continue
        s = train_df[col]
        kind = _column_ui_kind(s)
        if kind == "numeric":
            if _numeric_is_discrete_integers(s):
                st.session_state[key] = int(round(float(np.nanmedian(s.astype(float)))))
            else:
                st.session_state[key] = float(np.nanmedian(s.astype(float)))
        elif kind == "binary":
            u = sorted(s.dropna().unique(), key=lambda x: (str(type(x)), str(x)))
            mode = s.mode()
            pick = mode.iloc[0] if len(mode) else u[0]
            if isinstance(pick, (bool, np.bool_)):
                st.session_state[key] = bool(pick)
            elif dt_is_int_like(s) and not isinstance(pick, bool):
                st.session_state[key] = int(pick)
            else:
                st.session_state[key] = pick
        else:
            mode = s.mode()
            st.session_state[key] = mode.iloc[0] if len(mode) else sorted(s.astype(str).unique())[0]


def _render_numeric_slider(col: str, train_df: pd.DataFrame) -> None:
    s = train_df[col].dropna().astype(float)
    lo, hi = float(s.min()), float(s.max())
    if lo >= hi:
        lo, hi = lo - 1.0, hi + 1.0
    step = (hi - lo) / 200.0
    if step <= 0:
        step = 0.01
    key = _widget_key(col)
    if _numeric_is_discrete_integers(train_df[col]):
        lo_i, hi_i = int(np.floor(lo)), int(np.ceil(hi))
        st.slider(
            col,
            min_value=lo_i,
            max_value=max(hi_i, lo_i + 1),
            step=1,
            key=key,
            help=f"Training range ≈ [{lo_i}, {hi_i}]",
        )
    else:
        st.slider(
            col,
            min_value=lo,
            max_value=hi,
            step=step,
            format="%.4f",
            key=key,
            help=f"Training min/max: {lo:.4g} … {hi:.4g}",
        )


def _render_binary_select(col: str, train_df: pd.DataFrame) -> None:
    s = train_df[col].dropna()
    u = sorted(s.unique(), key=lambda x: (str(type(x)), str(x)))

    def _norm(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        return v

    opts = [_norm(x) for x in u]
    key = _widget_key(col)
    st.selectbox(col, options=opts, key=key)


def _render_categorical_select(col: str, train_df: pd.DataFrame) -> None:
    s = train_df[col].dropna().astype(str)
    opts = sorted(s.unique())
    key = _widget_key(col)
    st.selectbox(col, options=opts, key=key)


def _collect_features(required: list[str], train_df: pd.DataFrame) -> dict:
    out = {}
    for col in required:
        key = _widget_key(col)
        val = st.session_state[key]
        s = train_df[col] if not train_df.empty and col in train_df.columns else None
        kind = _column_ui_kind(s) if s is not None else "categorical"
        if kind == "numeric" and _numeric_is_discrete_integers(train_df[col]):
            out[col] = int(val)
        elif kind == "binary" and isinstance(val, (bool, np.bool_)):
            out[col] = bool(val)
        elif isinstance(val, (np.integer,)):
            out[col] = int(val)
        elif isinstance(val, (np.floating,)):
            out[col] = float(val)
        else:
            out[col] = val
    return out


def _fetch_prediction(client: httpx.Client, api_base: str, payload: dict):
    return client.post(f"{api_base.rstrip('/')}/predict", json=payload)

def _predict_class(
    client: httpx.Client,
    api_base: str,
    features: dict,
    *,
    include_shap: bool = False,
) -> tuple[float, int]:
    r = client.post(
        f"{api_base.rstrip('/')}/predict",
        json={"features": features, "include_shap": include_shap},
    )
    r.raise_for_status()
    j = r.json()
    return float(j["churn_probability"]), int(j["churn_binary"])


def _format_value(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def main():
    st.set_page_config(page_title="Churn prediction", layout="wide")
    st.title("Customer churn prediction")

    api_base = st.sidebar.text_input(
        "API base URL",
        value=DEFAULT_API,
        help="FastAPI server (e.g. http://127.0.0.1:8000)",
    )
    include_shap = st.sidebar.checkbox("Include SHAP explanations", value=True)
    timeout = 120.0

    train_df = _load_training_frame()

    with httpx.Client(timeout=timeout) as client:
        try:
            h = client.get(f"{api_base.rstrip('/')}/health")
            h.raise_for_status()
            st.sidebar.success("API reachable")
        except Exception as e:
            st.sidebar.error(f"API not reachable: {e}")
            st.info("Start the API: `PYTHONPATH=src uvicorn api.app:app --host 127.0.0.1 --port 8000`")
            return

        try:
            schema = client.get(f"{api_base.rstrip('/')}/schema").json()
        except Exception as e:
            st.error(f"Could not load /schema: {e}")
            return

        required = schema.get("required_columns", [])
        st.caption(
            f"Model: **{schema.get('model_name', '?')}** — **{len(required)}** raw features. "
            "Ranges and categories come from the training CSV. Adjust controls to refresh predictions."
        )

        if train_df.empty:
            st.warning("Training CSV not found; using neutral defaults for widget state.")

        _init_feature_defaults(required, train_df)

        features = _collect_features(required, train_df)
        payload = {"features": features, "include_shap": include_shap}

        if "last_prediction" not in st.session_state:
            st.session_state.last_prediction = None
        if "last_error" not in st.session_state:
            st.session_state.last_error = None

        try:
            r = _fetch_prediction(client, api_base, payload)
            r.raise_for_status()
            out = r.json()
            st.session_state.last_prediction = out
            st.session_state.last_error = None
        except Exception as e:
            st.session_state.last_error = str(e)
            out = st.session_state.last_prediction

        st.subheader("Prediction & explanations")
        if st.session_state.last_error and out is None:
            st.error(f"Prediction failed: {st.session_state.last_error}")
        elif out is None:
            st.warning("No prediction yet.")
        else:
            if st.session_state.last_error:
                st.caption(f"Could not refresh — showing last good result ({st.session_state.last_error})")

            c1, c2, c3 = st.columns(3)
            c1.metric("Churn probability", f"{out['churn_probability']:.4f}")
            c2.metric("Predicted class", "Churn" if out["churn_binary"] else "No churn")
            c3.metric("Decision threshold", f"{out['best_threshold']:.4f}")

            if include_shap and out.get("shap_contributions"):
                st.markdown("**SHAP contributions** (scaled feature space)")
                if out.get("expected_value") is not None:
                    st.caption(f"SHAP expected value (baseline): {out['expected_value']:.4f}")

                df = pd.DataFrame(out["shap_contributions"])
                top = df.head(20)
                fig = px.bar(
                    top,
                    x="shap",
                    y="feature",
                    orientation="h",
                    labels={"shap": "SHAP value", "feature": ""},
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=max(400, len(top) * 22))
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Counterfactuals (single-feature flips)**")
                st.caption(
                    "For the top 3 *raw input* features by |SHAP|, this tries to change just that one feature "
                    "within its training range until the predicted class flips (all other inputs fixed)."
                )

                required_set = set(required)
                raw_drivers = df[df["feature"].isin(required_set)].head(3)

                cf_rows = []
                base_binary = int(out["churn_binary"])

                for _, row in raw_drivers.iterrows():
                    feat = str(row["feature"])
                    cur_val = features.get(feat)

                    if train_df.empty or feat not in train_df.columns:
                        cf_rows.append(
                            {
                                "feature": feat,
                                "current": _format_value(cur_val),
                                "flip_to": "n/a",
                                "direction": "n/a",
                                "notes": "No training distribution available",
                            }
                        )
                        continue

                    s = train_df[feat]
                    kind = _column_ui_kind(s)

                    # Categorical/binary: try alternative categories/values
                    if kind in ("categorical", "binary"):
                        opts = (
                            sorted(s.dropna().astype(str).unique())
                            if kind == "categorical"
                            else sorted(s.dropna().unique(), key=lambda x: (str(type(x)), str(x)))
                        )
                        found = None
                        for opt in opts:
                            opt_val = opt
                            if kind == "binary":
                                if isinstance(opt_val, (np.integer, int)):
                                    opt_val = int(opt_val)
                                if isinstance(opt_val, (np.bool_, bool)):
                                    opt_val = bool(opt_val)
                            if str(opt_val) == str(cur_val):
                                continue
                            trial = dict(features)
                            trial[feat] = opt_val
                            try:
                                _, b = _predict_class(client, api_base, trial, include_shap=False)
                            except Exception:
                                continue
                            if b != base_binary:
                                found = opt_val
                                break

                        if found is None:
                            cf_rows.append(
                                {
                                    "feature": feat,
                                    "current": _format_value(cur_val),
                                    "flip_to": "no_flip",
                                    "direction": "n/a",
                                    "notes": "No single change among observed categories flips outcome",
                                }
                            )
                        else:
                            cf_rows.append(
                                {
                                    "feature": feat,
                                    "current": _format_value(cur_val),
                                    "flip_to": _format_value(found),
                                    "direction": "switch",
                                    "notes": "Switch category/value",
                                }
                            )
                        continue

                    # Numeric: probe min/max, then binary search on the side that flips
                    ss = s.dropna().astype(float)
                    lo, hi = float(ss.min()), float(ss.max())
                    try:
                        cur = float(cur_val)
                    except Exception:
                        cur = float(np.nanmedian(ss))

                    cur = float(np.clip(cur, lo, hi))

                    trial_min = dict(features)
                    trial_min[feat] = int(lo) if _numeric_is_discrete_integers(s) else lo
                    trial_max = dict(features)
                    trial_max[feat] = int(hi) if _numeric_is_discrete_integers(s) else hi

                    try:
                        _, b_min = _predict_class(client, api_base, trial_min, include_shap=False)
                        _, b_max = _predict_class(client, api_base, trial_max, include_shap=False)
                    except Exception as e:
                        cf_rows.append(
                            {
                                "feature": feat,
                                "current": _format_value(cur),
                                "flip_to": "n/a",
                                "direction": "n/a",
                                "notes": f"API error while probing bounds: {e}",
                            }
                        )
                        continue

                    if b_min == base_binary and b_max == base_binary:
                        cf_rows.append(
                            {
                                "feature": feat,
                                "current": _format_value(cur),
                                "flip_to": "no_flip",
                                "direction": "n/a",
                                "notes": "No flip within training min/max",
                            }
                        )
                        continue

                    if b_min != base_binary:
                        direction = "down"
                        a, b = lo, cur
                    else:
                        direction = "up"
                        a, b = cur, hi

                    # Binary search (approx) for the smallest change that flips
                    flip_at = None
                    for _ in range(14):
                        mid = (a + b) / 2.0
                        trial = dict(features)
                        trial[feat] = int(round(mid)) if _numeric_is_discrete_integers(s) else float(mid)
                        try:
                            _, pred_b = _predict_class(client, api_base, trial, include_shap=False)
                        except Exception:
                            break
                        if pred_b != base_binary:
                            flip_at = trial[feat]
                            b = mid
                        else:
                            a = mid

                    if flip_at is None:
                        cf_rows.append(
                            {
                                "feature": feat,
                                "current": _format_value(cur),
                                "flip_to": "n/a",
                                "direction": direction,
                                "notes": "Could not localize flip threshold",
                            }
                        )
                    else:
                        cf_rows.append(
                            {
                                "feature": feat,
                                "current": _format_value(cur),
                                "flip_to": _format_value(flip_at),
                                "direction": direction,
                                "notes": "Approx within training bounds",
                            }
                        )

                if len(raw_drivers) == 0:
                    st.info("Top SHAP drivers are engineered features; no raw-feature counterfactuals to show.")
                else:
                    st.dataframe(pd.DataFrame(cf_rows), use_container_width=True, hide_index=True)

                with st.expander("Full SHAP table"):
                    st.dataframe(df, use_container_width=True)

        st.subheader("Feature controls")
        w_left, w_right = st.columns(2)
        half = (len(required) + 1) // 2

        for i, col in enumerate(required):
            box = w_left if i < half else w_right
            with box:
                if train_df.empty or col not in train_df.columns:
                    st.number_input(col, key=_widget_key(col))
                    continue
                s = train_df[col]
                kind = _column_ui_kind(s)
                if kind == "numeric":
                    _render_numeric_slider(col, train_df)
                elif kind == "binary":
                    _render_binary_select(col, train_df)
                else:
                    _render_categorical_select(col, train_df)


if __name__ == "__main__":
    main()
