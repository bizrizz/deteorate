"""
SOFA-2–based deterioration label via clifpy (no re-implementation of SOFA logic).

Label_SOFA2(t) = 1 if (SOFA2_future_max_total - SOFA2_baseline_total) >= 2 else 0.
- Baseline: max SOFA-2 total over the previous 6 non-overlapping hourly windows (t-6h..t].
- Future max: max SOFA-2 total over the next 12 non-overlapping hourly windows (t..t+12h].

Windows for calculate_sofa2 must be non-overlapping per hospitalization_id; we use
one set of hourly windows (admission→admission+1h], (admission+1h→admission+2h], ...
and derive baseline/future from rolling max over those.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Optional: only required when calling calculate_sofa2
try:
    from clifpy import calculate_sofa2, SOFA2Config
except ImportError:
    calculate_sofa2 = None  # type: ignore
    SOFA2Config = None  # type: ignore


def _ensure_utc(df: pd.DataFrame, *cols: str) -> None:
    for c in cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            if df[c].dt.tz is None:
                df[c] = df[c].dt.tz_localize("UTC", ambiguous="infer")
        else:
            df[c] = pd.to_datetime(df[c], utc=True)


def build_hourly_cohort(
    hospitalization: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a cohort_df of non-overlapping 1-hour windows per hospitalization.

    For each stay, windows are (admission, admission+1h], (admission+1h, admission+2h], ...
    until the end of the window would be past discharge_dttm. No overlapping windows
    so that calculate_sofa2 can be called once.

    Returns:
        DataFrame with columns [hospitalization_id, start_dttm, end_dttm].
    """
    _ensure_utc(hospitalization, "admission_dttm", "discharge_dttm")
    rows = []
    for _, row in hospitalization.iterrows():
        hid = row["hospitalization_id"]
        start = row["admission_dttm"]
        end = row["discharge_dttm"]
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue
        # Non-overlapping hourly windows: (start, start+1h], (start+1h, start+2h], ...
        t_start = start
        while t_start < end:
            t_end = min(t_start + pd.Timedelta(hours=1), end)
            if t_end > t_start:
                rows.append({
                    "hospitalization_id": hid,
                    "start_dttm": t_start,
                    "end_dttm": t_end,
                })
            t_start = t_end
    return pd.DataFrame(rows)


def _rolling_max_6h(sofa_df: pd.DataFrame) -> pd.Series:
    """
    For each row (hour window), baseline_max_6h = max(sofa_total) over the current
    and previous 5 windows (6 windows ending at or before this hour's end).
    """
    g = sofa_df.sort_values(["hospitalization_id", "end_dttm"])
    return g.groupby("hospitalization_id", group_keys=False)["sofa_total"].transform(
        lambda s: s.rolling(window=6, min_periods=1).max()
    )


def _forward_max_12h(sofa_df: pd.DataFrame) -> pd.Series:
    """
    For each row (hour window), future_max_12h = max(sofa_total) over the NEXT 12
    windows (windows starting after this hour's end). NaN when there are fewer than
    12 future windows.
    """
    g = sofa_df.sort_values(["hospitalization_id", "end_dttm"])
    def fwd_max(s: pd.Series) -> pd.Series:
        v = s.values
        n = len(v)
        out = np.full(n, np.nan, dtype=float)
        for j in range(n):
            if j + 12 < n:
                out[j] = np.nanmax(v[j + 1 : j + 13])
        return pd.Series(out, index=s.index)
    return g.groupby("hospitalization_id", group_keys=False)["sofa_total"].transform(fwd_max)


def build_sofa2_labels_from_scores(
    sofa_df: pd.DataFrame,
    delta_threshold: int = 2,
) -> pd.DataFrame:
    """
    Derive deterioration label from SOFA-2 score DataFrame.

    Expects sofa_df to have columns: hospitalization_id, end_dttm, sofa_total
    (as returned by calculate_sofa2; start_dttm/end_dttm identify the window).

    Computes:
    - baseline_max_6h: rolling max of sofa_total over the 6 hourly windows ending at this hour.
    - future_max_12h: max of sofa_total over the next 12 hourly windows.
    - label: 1 if (future_max_12h - baseline_max_6h) >= delta_threshold else 0.
      Rows with missing future_max_12h (e.g. last 12h of stay) get label 0.

    Returns:
        DataFrame keyed by (hospitalization_id, hour_end_dttm) with
        baseline_max_6h, future_max_12h, delta_sofa2, label, and sofa_total.
    """
    # Normalize column name (clifpy may return sofa_total or similar)
    if "sofa_total" not in sofa_df.columns:
        cand = [c for c in sofa_df.columns if "sofa" in c.lower() and "total" in c.lower()]
        if not cand:
            raise ValueError("sofa_df must contain a SOFA total column (e.g. sofa_total)")
        sofa_df = sofa_df.rename(columns={cand[0]: "sofa_total"})

    sofa_df = sofa_df.sort_values(["hospitalization_id", "end_dttm"]).copy()
    sofa_df["baseline_max_6h"] = _rolling_max_6h(sofa_df)
    sofa_df["future_max_12h"] = _forward_max_12h(sofa_df)
    sofa_df["delta_sofa2"] = sofa_df["future_max_12h"] - sofa_df["baseline_max_6h"]
    sofa_df["label"] = (
        (sofa_df["delta_sofa2"].notna())
        & (sofa_df["delta_sofa2"] >= delta_threshold)
    ).astype(int)
    # Where future_max_12h is NaN (end of stay), set label to 0
    sofa_df.loc[sofa_df["future_max_12h"].isna(), "label"] = 0

    return sofa_df


def compute_sofa2_deterioration_labels(
    hospitalization: pd.DataFrame,
    clif_config_path: Optional[str] = None,
    sofa2_config: Optional["SOFA2Config"] = None,
    delta_threshold: int = 2,
) -> pd.DataFrame:
    """
    Full SOFA-2 labeling pipeline using clifpy.

    1) Build non-overlapping hourly cohort_df from hospitalization.
    2) Call calculate_sofa2(cohort_df, clif_config_path=..., sofa2_config=...).
    3) Derive baseline_max_6h, future_max_12h, label from rolling logic.

    Parameters
    ----------
    hospitalization : DataFrame
        Must have hospitalization_id, admission_dttm, discharge_dttm.
    clif_config_path : str, optional
        Path to CLIF config used by clifpy to load data. If None, uses env CLIF_CONFIG_PATH.
    sofa2_config : SOFA2Config, optional
        Passed to calculate_sofa2. Default SOFA2Config() if None.
    delta_threshold : int
        Label = 1 when (future_max_12h - baseline_max_6h) >= this value. Default 2.

    Returns
    -------
    DataFrame with columns:
        hospitalization_id, start_dttm, end_dttm (hour_end_dttm = end_dttm),
        baseline_max_6h, future_max_12h, delta_sofa2, label, sofa_total, plus any
        other columns returned by calculate_sofa2 (subscores, etc.).
    """
    if calculate_sofa2 is None:
        try:
            import clifpy as _clif
        except ImportError:
            raise ImportError(
                "SOFA-2 labels require clifpy. Install with: pip install clifpy"
            ) from None
        if getattr(_clif, "calculate_sofa2", None) is None:
            raise ImportError(
                "SOFA-2 labels need clifpy to export 'calculate_sofa2' (per-window SOFA-2). "
                "Your clifpy version does not provide it (it has compute_sofa_polars, which returns one row per hospitalization). "
                "Use event-based labeling (run without --sofa2) or a clifpy version that provides calculate_sofa2."
            )
        # clifpy is installed but top-level import of calculate_sofa2 failed for another reason
        raise ImportError("clifpy is required for SOFA-2 labels. Install with: pip install clifpy")

    import os
    config_path = clif_config_path or os.environ.get("CLIF_CONFIG_PATH")
    if not config_path or not Path(config_path).exists():
        raise FileNotFoundError(
            "clifpy requires clif_config_path (or env CLIF_CONFIG_PATH) pointing to an existing config file."
        )

    cohort_df = build_hourly_cohort(hospitalization)
    if cohort_df.empty:
        return pd.DataFrame(columns=[
            "hospitalization_id", "start_dttm", "end_dttm",
            "baseline_max_6h", "future_max_12h", "delta_sofa2", "label", "sofa_total",
        ])

    config = sofa2_config if sofa2_config is not None else SOFA2Config()
    sofa_df = calculate_sofa2(
        cohort_df=cohort_df,
        clif_config_path=config_path,
        return_rel=False,
        sofa2_config=config,
    )
    if sofa_df is None or sofa_df.empty:
        return build_sofa2_labels_from_scores(
            cohort_df.assign(sofa_total=np.nan),
            delta_threshold=delta_threshold,
        )

    # Preserve all cohort windows: left join cohort -> SOFA so missing SOFA rows get NaN
    merge_cols = ["hospitalization_id", "start_dttm", "end_dttm"]
    if all(c in sofa_df.columns for c in merge_cols):
        combined = cohort_df.merge(sofa_df, on=merge_cols, how="left")
    else:
        combined = sofa_df.copy()
    if "sofa_total" not in combined.columns:
        cand = [c for c in combined.columns if "sofa" in c.lower() and "total" in c.lower()]
        if cand:
            combined = combined.rename(columns={cand[0]: "sofa_total"})
    return build_sofa2_labels_from_scores(combined, delta_threshold=delta_threshold)


def get_sofa2_labeled_grid(
    sofa2_labels: pd.DataFrame,
    hospitalization: pd.DataFrame,
    min_hours_from_admission: int = 6,
) -> pd.DataFrame:
    """
    Produce a grid compatible with the rest of the pipeline: one row per
    (hospitalization_id, hour_dttm) with patient_id and label.

    - hour_dttm = end_dttm of the hourly window (hour_end_dttm).
    - Drops the first min_hours_from_admission hours so that 6h lookback is available.
    """
    if sofa2_labels.empty:
        return pd.DataFrame(columns=["hospitalization_id", "patient_id", "hour_dttm", "label"])
    _ensure_utc(hospitalization, "admission_dttm")
    grid = sofa2_labels[["hospitalization_id", "end_dttm", "baseline_max_6h", "future_max_12h", "label"]].copy()
    grid = grid.rename(columns={"end_dttm": "hour_dttm"})
    grid = grid.merge(
        hospitalization[["hospitalization_id", "patient_id", "admission_dttm"]],
        on="hospitalization_id",
        how="left",
    )
    grid["hours_since_admission"] = (
        grid["hour_dttm"] - grid["admission_dttm"]
    ).dt.total_seconds() / 3600.0
    grid = grid[grid["hours_since_admission"] >= min_hours_from_admission]
    grid = grid.drop(columns=["admission_dttm", "hours_since_admission"])
    return grid
