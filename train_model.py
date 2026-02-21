"""
Train XGBoost classifier for deterioration prediction.
Split by patient_id, evaluate AUROC and AUPRC, save model and compute SHAP values.
Full-dataset run: no development limits; diagnostics and memory check before training.
"""
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


# Default paths
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_FULL_RUN = DEFAULT_MODEL_DIR / "output_full_run"
DEFAULT_DATA_DIR = os.environ.get(
    "CLIF_DATA_DIR",
    "/Users/adambizios/Downloads/CLIF-MIMIC/Release 1.0.0 (MIMIC-IV 3.1 - CLIF 2",
)
WINDOWS_MEMORY_WARNING_THRESHOLD = 1_000_000


def _format_elapsed(seconds: float) -> str:
    """Format seconds as e.g. 1m 23.4s or 45.6s."""
    if seconds >= 60:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}m {s:.1f}s"
    return f"{seconds:.1f}s"


def _row_progress_bar(
    current: float,
    total: int,
    name: str,
    start_time: float,
) -> None:
    """Print a single line: [Name] [=====>    ] pct% (current/total) ETA: X. current may be float for fractional progress."""
    if total <= 0:
        return
    pct = min(100, int(100 * current / total))
    width = 20
    n_fill = min(width, int(width * current / total)) if total else 0
    incomplete = current < total
    n_eq = n_fill - (1 if incomplete and n_fill == width else 0)
    bar = "=" * n_eq + (">" if incomplete else "") + " " * (width - n_eq - (1 if incomplete else 0))
    elapsed = time.perf_counter() - start_time
    if current > 0 and current < total:
        rate = current / elapsed
        remaining = total - current
        eta_sec = remaining / rate
        eta_str = f"ETA: {_format_elapsed(eta_sec)}"
    elif current >= total:
        eta_str = "done"
    else:
        eta_str = "ETA: --"
    cur_disp = int(current) if current == int(current) else round(current, 1)
    line = f"\r[{name}] [{bar}] {pct}% ({cur_disp}/{total}) {eta_str}    "
    try:
        sys.stdout.write(line)
        sys.stdout.flush()
    except (BrokenPipeError, OSError):
        pass


def _indeterminate_bar(name: str, elapsed: float, step_durations: list, width: int = 20) -> str:
    """Build a progress bar for steps without row counts: fill by elapsed/avg_previous or half-fill."""
    if step_durations:
        avg_step = sum(step_durations) / len(step_durations)
        pct = min(1.0, elapsed / max(1e-9, avg_step))
        n_fill = min(width, int(width * pct))
    else:
        n_fill = width // 2  # indeterminate: half-filled
    n_fill = min(width, n_fill)
    bar = "=" * n_fill + (">" if n_fill < width else "") + " " * (width - n_fill - (1 if n_fill < width else 0))
    if step_durations:
        avg_step = sum(step_durations) / len(step_durations)
        remaining = max(0.0, avg_step - elapsed)
        eta_str = f"ETA: {_format_elapsed(remaining)}"
    else:
        eta_str = "ETA: --"
    return f"\r[{name}] [{bar}] elapsed {_format_elapsed(elapsed)} | {eta_str}    "


@contextmanager
def _realtime_step(
    name: str,
    step_durations: list,
    total_steps: int = 7,
):
    """
    Context manager: run a step with a real-time updating progress bar (elapsed + ETA).
    step_durations: list of previous step durations in seconds; we append this step's duration on exit.
    """
    start = time.perf_counter()
    stop = threading.Event()
    done = threading.Event()

    def _updater():
        while not stop.wait(1.0):
            elapsed = time.perf_counter() - start
            line = _indeterminate_bar(name, elapsed, step_durations)
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except (BrokenPipeError, OSError):
                break
        done.set()

    t = threading.Thread(target=_updater, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        done.wait(timeout=2.0)
        duration = time.perf_counter() - start
        step_durations.append(duration)
        try:
            sys.stdout.write(f"\r[{name}] [{'=' * 20}] {_format_elapsed(duration)} (done)    \n")
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            pass


class _XGBProgressCallback(xgb.callback.TrainingCallback):
    """XGBoost callback to drive a progress bar by iteration."""

    def __init__(self, progress_callback: Callable[[int, int], None], n_estimators: int):
        self._progress_callback = progress_callback
        self._n_estimators = n_estimators

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        if self._progress_callback is not None:
            self._progress_callback(min(epoch + 1, self._n_estimators), self._n_estimators)
        return False  # do not stop training


def get_feature_columns(X: pd.DataFrame) -> list[str]:
    """Exclude ids and datetime from feature list."""
    exclude = {"hospitalization_id", "hour_dttm", "patient_id"}
    return [c for c in X.columns if c not in exclude]


def train_test_split_by_patient(
    X: pd.DataFrame,
    y: pd.Series,
    patient_id: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split so all rows of a patient are in either train or test."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(
        splitter.split(X, y, groups=patient_id)
    )
    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
    )


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    patient_id: pd.Series,
    feature_cols: Optional[list[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    xgb_params: Optional[dict] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> tuple[xgb.XGBClassifier, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Train XGBoost with patient-based split. Returns model, X_train, X_test, y_train, y_test, feature_cols.
    If progress_callback(iteration, total) is given, it is called each boosting round.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(X)
    X_f = X[feature_cols].copy()
    for c in feature_cols:
        if X_f[c].dtype == object or (hasattr(X_f[c].dtype, "name") and X_f[c].dtype.name == "category"):
            X_f[c] = pd.to_numeric(X_f[c], errors="coerce")
    X_f = X_f.fillna(np.nan)  # XGBoost handles NaN
    y = y.astype(int).values

    X_train, X_test, y_train, y_test = train_test_split_by_patient(
        X_f, pd.Series(y), patient_id, test_size=test_size, random_state=random_state
    )
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "use_label_encoder": False,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if xgb_params:
        params.update(xgb_params)
    n_estimators = params.get("n_estimators", 100)
    model = xgb.XGBClassifier(**params)
    fit_kwargs = dict(
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    if progress_callback is not None:
        try:
            callbacks = [_XGBProgressCallback(progress_callback, n_estimators)]
            model.fit(X_train, y_train, **fit_kwargs, callbacks=callbacks)
        except (TypeError, AttributeError, ValueError):
            # XGBoost sklearn API may not support callbacks in this version
            model.fit(X_train, y_train, **fit_kwargs)
    else:
        model.fit(X_train, y_train, **fit_kwargs)
    return model, X_train, X_test, pd.Series(y_train), pd.Series(y_test), feature_cols


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list[str],
) -> dict:
    """Compute AUROC and AUPRC."""
    X_f = X_test[feature_cols]
    proba = model.predict_proba(X_f)[:, 1]
    return {
        "auroc": float(roc_auc_score(y_test, proba)),
        "auprc": float(average_precision_score(y_test, proba)),
    }


def compute_shap(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    feature_cols: list[str],
    max_samples: Optional[int] = 1000,
) -> tuple[np.ndarray, shap.Explanation]:
    """
    Compute SHAP values for feature importance. Uses a sample if X is large.
    Returns (shap_values, explanation).
    """
    X_f = X[feature_cols].copy()
    # TreeExplainer requires float64; object/Int64 dtypes cause "Cannot cast... to float64"
    for c in X_f.columns:
        X_f[c] = pd.to_numeric(X_f[c], errors="coerce")
    X_f = X_f.astype(np.float64, copy=False)
    X_f = X_f.fillna(0.0)
    if max_samples is not None and len(X_f) > max_samples:
        X_f = X_f.sample(n=max_samples, random_state=42)
    explainer = shap.TreeExplainer(model, X_f, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_f)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class
    return shap_values, explainer


def save_model_and_artifacts(
    model: xgb.XGBClassifier,
    feature_cols: list[str],
    metrics: dict,
    shap_values: Optional[np.ndarray] = None,
    feature_names: Optional[list[str]] = None,
    out_dir: Optional[os.PathLike] = None,
) -> None:
    """Save model (PKL + JSON), feature_list.json, metrics.json, shap_importance.csv."""
    out_dir = Path(out_dir or DEFAULT_MODEL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.pkl")
    model.save_model(str(out_dir / "deterioration_xgb_model.json"))
    with open(out_dir / "feature_list.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    if shap_values is not None and feature_names is not None:
        mean_abs = np.abs(shap_values).mean(axis=0)
        order = np.argsort(-mean_abs)
        shap_df = pd.DataFrame({
            "feature": [feature_names[i] for i in order],
            "mean_abs_shap": [float(mean_abs[i]) for i in order],
        })
        shap_df.to_csv(out_dir / "shap_importance.csv", index=False)


def run_pipeline(
    data_dir: Optional[str] = None,
    out_dir: Optional[os.PathLike] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    shap_sample: Optional[int] = 1000,
    use_sofa2_label: bool = False,
    clif_config_path: Optional[str] = None,
    sofa2_delta_threshold: int = 2,
    max_windows: Optional[int] = None,
    max_rows_per_table: Optional[int] = None,
) -> dict:
    """
    Full pipeline: load data, label, build features, train, evaluate, SHAP, save.
    Returns metrics dict.

    If use_sofa2_label is True, deterioration is defined by ΔSOFA-2 ≥ sofa2_delta_threshold
    (via clifpy); requires clif_config_path (or env CLIF_CONFIG_PATH). Otherwise uses
    event-based label (death, new vasopressor/IMV/CRRT/ECMO in next 12h).

    If max_windows is set and the labeled grid is larger, a stratified subsample of that
    many hourly windows is used for features and training (reduces RAM and runtime).

    If max_rows_per_table is set, each parquet table is capped at that many rows when
    loading (chunked read) to avoid OOM on memory-limited machines.
    """
    from data_extraction import load_all_tables
    from feature_engineering import build_all_features
    from labeling import create_labeled_grid

    data_dir = data_dir or DEFAULT_DATA_DIR
    pipeline_start = time.perf_counter()
    step_durations: list[float] = []

    with _realtime_step("Load tables", step_durations):
        tables = load_all_tables(
            data_dir=data_dir,
            include_intake_output=True,
            max_rows_per_table=max_rows_per_table,
        )

    hospitalization = tables["hospitalization"]
    patient = tables["patient"]
    med_cont = tables["medication_admin_continuous"]
    respiratory = tables["respiratory_support"]
    crrt = tables["crrt_therapy"]
    ecmo = tables["ecmo_mcs"]
    vitals = tables["vitals"]
    labs = tables["labs"]
    intake_output = tables.get("intake_output")

    # Labeling: row-based progress (hospitalizations) for event-based path; else elapsed-only
    step_start = time.perf_counter()
    if use_sofa2_label:
        with _realtime_step("Labeling grid", step_durations):
            from labeling_sofa2 import compute_sofa2_deterioration_labels, get_sofa2_labeled_grid
            sofa2_labels = compute_sofa2_deterioration_labels(
                hospitalization,
                clif_config_path=clif_config_path,
                sofa2_config=None,
                delta_threshold=sofa2_delta_threshold,
            )
            grid = get_sofa2_labeled_grid(
                sofa2_labels,
                hospitalization,
                min_hours_from_admission=6,
            )
    else:
        def _label_progress(completed: int, total: int) -> None:
            _row_progress_bar(completed, total, "Labeling grid", step_start)
        try:
            grid = create_labeled_grid(
                hospitalization,
                patient,
                med_cont,
                respiratory,
                crrt,
                ecmo,
                min_hours_from_admission=6,
                progress_callback=_label_progress,
            )
        finally:
            step_durations.append(time.perf_counter() - step_start)
            try:
                sys.stdout.write(f"\r[Labeling grid] {_format_elapsed(step_durations[-1])} (done)    \n")
                sys.stdout.flush()
            except (BrokenPipeError, OSError):
                pass
    if grid.empty:
        raise ValueError("Labeled grid is empty. Check data and date ranges.")

    # Optional cap: stratified subsample to reduce RAM and runtime
    if max_windows is not None and len(grid) > max_windows:
        with _realtime_step("Subsample", step_durations):
            frac = max_windows / len(grid)
            grid, _ = train_test_split(
                grid,
                train_size=frac,
                stratify=grid["label"],
                random_state=random_state,
            )
            grid = grid.reset_index(drop=True)
        print(f"  Subsampled to {len(grid):,} windows (--max-windows={max_windows:,}, stratified by label).")

    # Dataset diagnostics (full cohort, no limits)
    n_hospitalizations = grid["hospitalization_id"].nunique()
    n_hourly_windows = len(grid)
    label_positive_rate = grid["label"].mean()
    n_unique_patients = grid["patient_id"].nunique()
    print("--- Dataset diagnostics ---")
    print(f"  Total hospitalizations:    {n_hospitalizations}")
    print(f"  Total hourly windows:     {n_hourly_windows}")
    print(f"  Label positive rate:      {label_positive_rate:.4f}")
    print(f"  Unique patient_id:        {n_unique_patients}")
    if n_hourly_windows > WINDOWS_MEMORY_WARNING_THRESHOLD:
        print(f"\n  *** MEMORY WARNING: {n_hourly_windows} hourly windows (> {WINDOWS_MEMORY_WARNING_THRESHOLD:,}). "
              "Training may require substantial RAM. ***\n")

    # Features: row-based progress (5 blocks)
    step_start = time.perf_counter()
    def _feat_progress(completed: int, total: int) -> None:
        _row_progress_bar(completed, total, "Features", step_start)
    try:
        X = build_all_features(
            grid,
            vitals,
            labs,
            med_cont,
            respiratory,
            intake_output,
            lookback_hours=6.0,
            progress_callback=_feat_progress,
        )
    finally:
        step_durations.append(time.perf_counter() - step_start)
        try:
            sys.stdout.write(f"\r[Features] {_format_elapsed(step_durations[-1])} (done)    \n")
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            pass
    # Merge labels from grid (one row per hour per hosp)
    grid_unique = grid[["hospitalization_id", "hour_dttm", "label"]].drop_duplicates()
    X = X.merge(
        grid_unique,
        on=["hospitalization_id", "hour_dttm"],
        how="inner",
    )
    y = X["label"].astype(int)
    X = X.drop(columns=["label"])
    patient_id = X["patient_id"]

    feature_cols = get_feature_columns(X)
    if not feature_cols:
        raise ValueError("No feature columns found. Check feature_engineering output.")
    X[feature_cols] = X[feature_cols].apply(pd.to_numeric, errors="coerce")

    n_rows, n_feat = len(X), len(feature_cols)
    print(f"  Estimated feature matrix shape: ({n_rows}, {n_feat})")
    print("-----------------------------------")
    print("Train/test split by patient_id (GroupShuffleSplit; no row-level shuffle).")

    # Train XGBoost with iteration-based progress bar
    step_start = time.perf_counter()
    def _train_progress(iteration: int, total: int) -> None:
        _row_progress_bar(iteration, total, "Train XGBoost", step_start)
    try:
        model, X_train, X_test, y_train, y_test, feature_cols_used = train_model(
            X,
            y,
            patient_id,
            feature_cols=feature_cols,
            test_size=test_size,
            random_state=random_state,
            progress_callback=_train_progress,
        )
    finally:
        step_durations.append(time.perf_counter() - step_start)
        try:
            sys.stdout.write(f"\r[Train XGBoost] [{'=' * 20}] {_format_elapsed(step_durations[-1])} (done)    \n")
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            pass

    with _realtime_step("Evaluate", step_durations):
        metrics = evaluate_model(model, X_test, y_test, feature_cols_used)

    with _realtime_step("SHAP", step_durations):
        shap_values, _ = compute_shap(
            model,
            pd.concat([X_train, X_test], ignore_index=True),
            feature_cols_used,
            max_samples=shap_sample,
        )

    with _realtime_step("Save artifacts", step_durations):
        save_model_and_artifacts(
            model,
            feature_cols_used,
            metrics,
            shap_values=shap_values,
            feature_names=feature_cols_used,
            out_dir=out_dir,
        )

    # Window-level predictions table (for API / Lovable)
    out_path = Path(out_dir or DEFAULT_OUTPUT_FULL_RUN)
    out_path.mkdir(parents=True, exist_ok=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(
        splitter.split(X[feature_cols_used], y, groups=patient_id)
    )
    risk_score = model.predict_proba(X[feature_cols_used])[:, 1]
    pred_df = pd.DataFrame({
        "hospitalization_id": X["hospitalization_id"].astype(str),
        "patient_id": X["patient_id"].astype(str),
        "timestamp": X["hour_dttm"],
        "risk_score": risk_score.astype(float),
        "label": y.values.astype(int),
    })
    pred_df["split"] = "test"
    pred_df.loc[train_idx, "split"] = "train"
    pred_df["start_dttm"] = pred_df["timestamp"] - pd.Timedelta(hours=1)
    pred_df["end_dttm"] = pred_df["timestamp"]
    pred_df.to_parquet(out_path / "predictions.parquet", index=False)

    # Final summary
    total_elapsed = time.perf_counter() - pipeline_start
    print("\n--- Final summary ---")
    print(f"  Training rows:  {len(X_train)}")
    print(f"  Test rows:     {len(X_test)}")
    print(f"  AUROC:         {metrics['auroc']:.4f}")
    print(f"  AUPRC:         {metrics['auprc']:.4f}")
    print(f"  Total time:    {_format_elapsed(total_elapsed)}")
    print("--------------------")
    return metrics


if __name__ == "__main__":
    run_pipeline(
        data_dir=DEFAULT_DATA_DIR,
        out_dir=DEFAULT_OUTPUT_FULL_RUN,
        test_size=0.2,
        random_state=42,
        shap_sample=10000,
        use_sofa2_label=False,
    )
