"""
FastAPI service for deterioration predictions and explainability.
Serves predictions table and per-window SHAP + timeseries for Lovable.
"""
import os
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Artifacts and data dirs (env or default)
ARTIFACTS_DIR = Path(
    os.environ.get("DETECTORATE_ARTIFACTS_DIR", str(Path(__file__).resolve().parent / "output_full_run"))
)
CLIF_DATA_DIR = os.environ.get("CLIF_DATA_DIR", "")
# Cap rows per table when loading CLIF for /explain (avoids OOM). Default 5M; set to 0 for no limit.
_clif_max_rows = os.environ.get("CLIF_MAX_ROWS_PER_TABLE", "")
if _clif_max_rows == "0":
    CLIF_MAX_ROWS_PER_TABLE = None
elif _clif_max_rows.isdigit():
    CLIF_MAX_ROWS_PER_TABLE = int(_clif_max_rows)
else:
    CLIF_MAX_ROWS_PER_TABLE = 5_000_000

app = FastAPI(title="ICU Deterioration API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded lazily (or at startup if all files present)
_model = None
_feature_cols: list[str] = []
_predictions_df: Optional[pd.DataFrame] = None
_tables_cache: Optional[dict] = None
_artifacts_error: Optional[str] = None  # Set when load fails so we can return 503 with message


def _ensure_utc(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        if series.dt.tz is None:
            return series.dt.tz_localize("UTC", ambiguous="infer")
    return pd.to_datetime(series, utc=True)


def _load_artifacts() -> None:
    global _model, _feature_cols, _predictions_df, _artifacts_error
    if _model is not None:
        return
    if _artifacts_error is not None:
        return
    model_path = ARTIFACTS_DIR / "model.pkl"
    feature_path = ARTIFACTS_DIR / "feature_list.json"
    pred_path = ARTIFACTS_DIR / "predictions.parquet"
    if not model_path.exists():
        _artifacts_error = f"Model not found: {model_path}. Run the pipeline and include output_full_run in the image."
        return
    if not feature_path.exists():
        _artifacts_error = f"Feature list not found: {feature_path}"
        return
    if not pred_path.exists():
        _artifacts_error = (
            f"Predictions not found: {pred_path}. Run the pipeline and add output_full_run/predictions.parquet to the image."
        )
        return
    try:
        _model = joblib.load(model_path)
        import json
        with open(feature_path) as f:
            _feature_cols = json.load(f)
        _predictions_df = pd.read_parquet(pred_path)
    except Exception as e:
        _artifacts_error = f"Failed to load artifacts: {e}"
        return
    # Normalize timestamp for lookups
    if _predictions_df["timestamp"].dt.tz is None:
        _predictions_df["timestamp"] = _predictions_df["timestamp"].dt.tz_localize("UTC", ambiguous="infer")
    if "hospitalization_id" in _predictions_df.columns and _predictions_df["hospitalization_id"].dtype != str:
        _predictions_df["hospitalization_id"] = _predictions_df["hospitalization_id"].astype(str)
    if "patient_id" in _predictions_df.columns and _predictions_df["patient_id"].dtype != str:
        _predictions_df["patient_id"] = _predictions_df["patient_id"].astype(str)


def _load_tables() -> dict:
    global _tables_cache
    if _tables_cache is not None:
        return _tables_cache
    if not CLIF_DATA_DIR or not Path(CLIF_DATA_DIR).is_dir():
        raise HTTPException(
            status_code=503,
            detail="CLIF_DATA_DIR not set or invalid. Set it to the CLIF parquet directory for /explain.",
        )
    from data_extraction import load_all_tables
    _tables_cache = load_all_tables(
        data_dir=CLIF_DATA_DIR,
        include_intake_output=True,
        max_rows_per_table=CLIF_MAX_ROWS_PER_TABLE,
    )
    return _tables_cache


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness/readiness: app is up. Does not load artifacts (so deploy succeeds even if parquet is missing)."""
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    """Base URL for Lovable / health check."""
    return {"service": "ICU Deterioration API", "docs": "/docs", "patients": "/patients", "explain": "/explain"}


@app.get("/patients")
def list_patients(
    limit: int = Query(50, ge=1, le=500),
    search: Optional[str] = Query(None),
) -> list[dict[str, Any]]:
    """List hospitalizations with latest timestamp and risk_score. Optional search by hospitalization_id or patient_id."""
    _load_artifacts()
    if _predictions_df is None:
        raise HTTPException(status_code=503, detail=_artifacts_error or "Artifacts not loaded.")
    df = _predictions_df.copy()
    # Latest row per hospitalization (by timestamp)
    idx = df.groupby("hospitalization_id")["timestamp"].idxmax()
    latest = df.loc[idx]
    if search and search.strip():
        q = search.strip().lower()
        mask = (
            latest["hospitalization_id"].astype(str).str.lower().str.contains(q, na=False)
            | latest["patient_id"].astype(str).str.lower().str.contains(q, na=False)
        )
        latest = latest[mask]
    latest = latest.sort_values("risk_score", ascending=False).head(limit)
    return [
        {
            "hospitalization_id": str(row["hospitalization_id"]),
            "patient_id": str(row["patient_id"]),
            "latest_timestamp": row["timestamp"].isoformat() if pd.notna(row["timestamp"]) else None,
            "risk_score": float(row["risk_score"]),
        }
        for _, row in latest.iterrows()
    ]


def _get_timeseries_for_window(
    hospitalization_id: str,
    end_ts: pd.Timestamp,
    lookback_hours: float = 6.0,
) -> dict[str, list[dict[str, Any]]]:
    """Pull last 6h of hr, map, spo2, fio2, peep, lactate, creatinine for one hospitalization."""
    tables = _load_tables()
    t_start = end_ts - pd.Timedelta(hours=lookback_hours)
    out: dict[str, list[dict[str, Any]]] = {
        "hr": [], "map": [], "spo2": [], "fio2": [], "peep": [], "lactate": [], "creatinine": [],
    }
    vitals = tables.get("vitals")
    if vitals is not None and not vitals.empty:
        vitals = vitals[vitals["hospitalization_id"].astype(str) == str(hospitalization_id)].copy()
        time_col = "recorded_dttm" if "recorded_dttm" in vitals.columns else None
        if time_col is None:
            cand = [c for c in vitals.columns if "dttm" in c.lower() or "time" in c.lower()]
            time_col = cand[0] if cand else None
        if time_col:
            vitals[time_col] = _ensure_utc(vitals[time_col])
            vitals = vitals[(vitals[time_col] > t_start) & (vitals[time_col] <= end_ts)]
        cat_col = "vital_category" if "vital_category" in vitals.columns else "vital_category"
        val_col = "vital_value" if "vital_value" in vitals.columns else None
        if val_col is None:
            val_col = "value" if "value" in vitals.columns else None
        if time_col and cat_col in vitals.columns and val_col in vitals.columns:
            for cat, key in [("heart_rate", "hr"), ("map", "map"), ("spo2", "spo2")]:
                sub = vitals[vitals[cat_col].astype(str).str.lower() == cat]
                for _, r in sub.iterrows():
                    t = r[time_col]
                    try:
                        v = float(pd.to_numeric(r[val_col], errors="coerce"))
                    except (TypeError, ValueError):
                        continue
                    if pd.notna(v):
                        out[key].append({"t": t.isoformat() if hasattr(t, "isoformat") else str(t), "v": v})
                out[key].sort(key=lambda x: x["t"])
    labs = tables.get("labs")
    if labs is not None and not labs.empty:
        labs = labs[labs["hospitalization_id"].astype(str) == str(hospitalization_id)].copy()
        time_col = "lab_result_dttm" if "lab_result_dttm" in labs.columns else "recorded_dttm"
        if time_col not in labs.columns:
            time_col = [c for c in labs.columns if "dttm" in c.lower()][0] if any("dttm" in c.lower() for c in labs.columns) else None
        if time_col:
            labs[time_col] = _ensure_utc(labs[time_col])
            labs = labs[(labs[time_col] > t_start) & (labs[time_col] <= end_ts)]
        cat_col = "lab_category" if "lab_category" in labs.columns else "lab_category"
        val_col = "lab_value" if "lab_value" in labs.columns else "lab_value_numeric"
        if val_col not in labs.columns:
            val_col = "lab_value"
        if cat_col in labs.columns:
            for cat, key in [("lactate", "lactate"), ("creatinine", "creatinine")]:
                sub = labs[labs[cat_col].astype(str).str.lower() == cat]
                for _, r in sub.iterrows():
                    t = r[time_col]
                    try:
                        v = float(pd.to_numeric(r[val_col], errors="coerce"))
                    except (TypeError, ValueError):
                        continue
                    if pd.notna(v):
                        out[key].append({"t": t.isoformat() if hasattr(t, "isoformat") else str(t), "v": v})
                out[key].sort(key=lambda x: x["t"])
    resp = tables.get("respiratory_support")
    if resp is not None and not resp.empty:
        resp = resp[resp["hospitalization_id"].astype(str) == str(hospitalization_id)].copy()
        time_col = "recorded_dttm" if "recorded_dttm" in resp.columns else None
        if time_col:
            resp[time_col] = _ensure_utc(resp[time_col])
            resp = resp[(resp[time_col] > t_start) & (resp[time_col] <= end_ts)]
            for col, key in [("fio2_set", "fio2"), ("peep_set", "peep")]:
                if col in resp.columns:
                    for _, r in resp.iterrows():
                        t = r[time_col]
                        try:
                            v = float(pd.to_numeric(r[col], errors="coerce"))
                        except (TypeError, ValueError):
                            continue
                        if pd.notna(v):
                            out[key].append({"t": t.isoformat() if hasattr(t, "isoformat") else str(t), "v": v})
                    out[key].sort(key=lambda x: x["t"])
    return out


def _empty_timeseries() -> dict[str, list]:
    return {"hr": [], "map": [], "spo2": [], "fio2": [], "peep": [], "lactate": [], "creatinine": []}


@app.get("/explain")
def explain(
    hospitalization_id: str = Query(..., description="Hospitalization ID"),
    timestamp: str = Query(..., description="Window end time (ISO)"),
) -> dict[str, Any]:
    """
    Explain one window: risk_score and label from predictions; if CLIF_DATA_DIR is set,
    also returns top_drivers (SHAP) and timeseries. Without CLIF, returns empty drivers/timeseries.
    """
    _load_artifacts()
    if _predictions_df is None:
        raise HTTPException(status_code=503, detail=_artifacts_error or "Artifacts not loaded.")
    try:
        ts = pd.to_datetime(timestamp, utc=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid timestamp: {e}")
    pred = _predictions_df[
        (_predictions_df["hospitalization_id"].astype(str) == str(hospitalization_id))
        & (_predictions_df["timestamp"] == ts)
    ]
    if pred.empty:
        ts_floor = ts.floor("h") if hasattr(ts, "floor") else pd.Timestamp(ts).floor("h")
        pred = _predictions_df[
            (_predictions_df["hospitalization_id"].astype(str) == str(hospitalization_id))
            & (pd.to_datetime(_predictions_df["timestamp"], utc=True).dt.floor("h") == ts_floor)
        ]
    if pred.empty:
        raise HTTPException(status_code=404, detail=f"No prediction for hospitalization_id={hospitalization_id}, timestamp={timestamp}")
    row = pred.iloc[0]
    patient_id = str(row["patient_id"])
    risk_score = float(row["risk_score"])
    label = int(row["label"])

    # If CLIF not available, return prediction only (no SHAP/timeseries)
    if not CLIF_DATA_DIR or not Path(CLIF_DATA_DIR).is_dir():
        return {
            "hospitalization_id": hospitalization_id,
            "patient_id": patient_id,
            "timestamp": ts.isoformat(),
            "risk_score": risk_score,
            "label": label,
            "top_drivers": [],
            "timeseries": _empty_timeseries(),
            "explain_available": False,
            "message": "SHAP and timeseries require CLIF_DATA_DIR. Run API locally with CLIF data for full explain.",
        }

    try:
        tables = _load_tables()
    except HTTPException:
        return {
            "hospitalization_id": hospitalization_id,
            "patient_id": patient_id,
            "timestamp": ts.isoformat(),
            "risk_score": risk_score,
            "label": label,
            "top_drivers": [],
            "timeseries": _empty_timeseries(),
            "explain_available": False,
            "message": "CLIF data could not be loaded. Risk and label from predictions only.",
        }

    from feature_engineering import build_all_features

    grid = pd.DataFrame([{
        "hospitalization_id": hospitalization_id,
        "hour_dttm": ts,
        "patient_id": patient_id,
    }])
    X_row = build_all_features(
        grid,
        tables["vitals"],
        tables["labs"],
        tables["medication_admin_continuous"],
        tables["respiratory_support"],
        tables.get("intake_output"),
        lookback_hours=6.0,
    )
    if X_row.empty:
        return {
            "hospitalization_id": hospitalization_id,
            "patient_id": patient_id,
            "timestamp": ts.isoformat(),
            "risk_score": risk_score,
            "label": label,
            "top_drivers": [],
            "timeseries": _empty_timeseries(),
            "explain_available": False,
            "message": "No feature data in lookback window.",
        }
    for c in _feature_cols:
        if c not in X_row.columns:
            X_row[c] = np.nan
    X_f = X_row[_feature_cols].copy()
    for c in _feature_cols:
        X_f[c] = pd.to_numeric(X_f[c], errors="coerce")
    X_f = X_f.astype(np.float64).fillna(0.0)

    explainer = shap.TreeExplainer(_model, X_f, feature_perturbation="interventional")
    shap_vals = explainer.shap_values(X_f)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_row = np.asarray(shap_vals).flatten()

    order = np.argsort(-np.abs(shap_row))[:15]
    top_drivers = []
    for i in order:
        if i >= len(_feature_cols):
            continue
        feat = _feature_cols[i]
        val = float(X_f.iloc[0].iloc[i])
        s = float(shap_row[i])
        direction = "increases_risk" if s > 0 else "decreases_risk" if s < 0 else "neutral"
        top_drivers.append({"feature": feat, "value": val, "shap": round(s, 6), "direction": direction})

    timeseries = _get_timeseries_for_window(hospitalization_id, ts, lookback_hours=6.0)

    return {
        "hospitalization_id": hospitalization_id,
        "patient_id": patient_id,
        "timestamp": ts.isoformat(),
        "risk_score": risk_score,
        "label": label,
        "top_drivers": top_drivers,
        "timeseries": timeseries,
        "explain_available": True,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
