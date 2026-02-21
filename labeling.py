"""
Build hourly time grid per hospitalization and compute binary deterioration label.
Label = 1 if within the NEXT 12 hours: death, new vasopressor, new IMV, CRRT start, ECMO start.
"""
from typing import Callable, Iterator, Optional

import pandas as pd
import numpy as np


def _ensure_utc(df: pd.DataFrame, *cols: str) -> None:
    for c in cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            if df[c].dt.tz is None:
                df[c] = df[c].dt.tz_localize("UTC", ambiguous="infer")
        else:
            df[c] = pd.to_datetime(df[c], utc=True)


def _build_hourly_grid_chunk(
    hosp: pd.DataFrame,
    admission_dttm: pd.Series,
    discharge_dttm: pd.Series,
) -> pd.DataFrame:
    """Vectorized: build hourly grid for one chunk. admission_dttm, discharge_dttm already UTC."""
    start_floor = admission_dttm.dt.floor("h")
    # Number of full hours with t < end
    n_hours = (
        (discharge_dttm - start_floor).dt.total_seconds().div(3600).astype(int).clip(lower=0)
    )
    valid = n_hours > 0
    if not valid.any():
        return pd.DataFrame(columns=["hospitalization_id", "patient_id", "hour_dttm"])
    hosp = hosp.loc[valid].reset_index(drop=True)
    n_hours = n_hours.loc[valid].values
    start_floor = start_floor.loc[valid]
    total_rows = int(n_hours.sum())
    # Repeat row index for each hour
    idx = np.repeat(np.arange(len(hosp)), n_hours)
    # Hour offset within each stay (0, 1, ..., n-1) without Python loop
    group_start = np.cumsum(np.concatenate([[0], n_hours[:-1]]))
    group_id = np.searchsorted(group_start, np.arange(total_rows), side="right") - 1
    offsets = np.arange(total_rows) - group_start[group_id]
    # Build hour_dttm: start_floor + offset hours (vectorized, preserves tz)
    hour_dttm = start_floor.iloc[idx].reset_index(drop=True) + pd.to_timedelta(offsets, unit="h")
    grid = pd.DataFrame(
        {
            "hospitalization_id": hosp["hospitalization_id"].values[idx],
            "patient_id": hosp["patient_id"].values[idx],
            "hour_dttm": hour_dttm,
        }
    )
    return grid


def build_hourly_grid(
    hospitalization: pd.DataFrame,
    min_hours_from_admission: int = 6,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    chunk_size: Optional[int] = 50_000,
) -> pd.DataFrame:
    """
    For each hospitalization, generate hourly timestamps from admission to discharge.
    Optionally skip first min_hours_from_admission hours (no 6h lookback yet).
    Uses vectorized chunked processing; progress_callback(completed, total) is called per chunk.
    Set chunk_size=None to process in one vectorized pass (no progress during build).
    """
    _ensure_utc(hospitalization, "admission_dttm", "discharge_dttm")
    hosp = hospitalization[["hospitalization_id", "patient_id", "admission_dttm", "discharge_dttm"]].copy()
    # Drop invalid rows once
    valid = (
        hosp["admission_dttm"].notna()
        & hosp["discharge_dttm"].notna()
        & (hosp["discharge_dttm"] > hosp["admission_dttm"])
    )
    hosp = hosp.loc[valid].reset_index(drop=True)
    total = len(hosp)
    if total == 0:
        return pd.DataFrame(columns=["hospitalization_id", "patient_id", "hour_dttm"])

    if chunk_size is None or chunk_size >= total:
        # Single vectorized pass
        grid = _build_hourly_grid_chunk(
            hosp,
            hosp["admission_dttm"],
            hosp["discharge_dttm"],
        )
        if progress_callback is not None:
            progress_callback(total, total)
    else:
        chunks = []
        n_chunks = (total + chunk_size - 1) // chunk_size
        for k in range(n_chunks):
            lo = k * chunk_size
            hi = min(lo + chunk_size, total)
            chunk_hosp = hosp.iloc[lo:hi]
            chunk_grid = _build_hourly_grid_chunk(
                chunk_hosp,
                chunk_hosp["admission_dttm"],
                chunk_hosp["discharge_dttm"],
            )
            if not chunk_grid.empty:
                chunks.append(chunk_grid)
            if progress_callback is not None:
                progress_callback(hi, total)
        grid = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["hospitalization_id", "patient_id", "hour_dttm"])

    if grid.empty:
        return grid
    # Drop hours that don't have full 6h lookback
    grid = grid.merge(
        hospitalization[["hospitalization_id", "admission_dttm"]],
        on="hospitalization_id",
        how="left",
    )
    grid["hours_since_admission"] = (
        grid["hour_dttm"] - grid["admission_dttm"]
    ).dt.total_seconds() / 3600.0
    grid = grid[grid["hours_since_admission"] >= min_hours_from_admission].drop(
        columns=["admission_dttm"]
    )
    return grid


def iter_hourly_grid_chunks(
    hospitalization: pd.DataFrame,
    min_hours_from_admission: int = 6,
    chunk_size: int = 50_000,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Iterator[pd.DataFrame]:
    """
    Yield hourly grid chunks without ever holding the full grid in memory.
    Each yielded chunk is filtered by min_hours_from_admission.
    Use this with add_deterioration_label per chunk to reduce peak memory.
    """
    _ensure_utc(hospitalization, "admission_dttm", "discharge_dttm")
    hosp = hospitalization[["hospitalization_id", "patient_id", "admission_dttm", "discharge_dttm"]].copy()
    valid = (
        hosp["admission_dttm"].notna()
        & hosp["discharge_dttm"].notna()
        & (hosp["discharge_dttm"] > hosp["admission_dttm"])
    )
    hosp = hosp.loc[valid].reset_index(drop=True)
    total = len(hosp)
    if total == 0:
        return
    n_chunks = (total + chunk_size - 1) // chunk_size
    for k in range(n_chunks):
        lo = k * chunk_size
        hi = min(lo + chunk_size, total)
        chunk_hosp = hosp.iloc[lo:hi]
        chunk_grid = _build_hourly_grid_chunk(
            chunk_hosp,
            chunk_hosp["admission_dttm"],
            chunk_hosp["discharge_dttm"],
        )
        if chunk_grid.empty:
            if progress_callback is not None:
                progress_callback(hi, total)
            continue
        chunk_grid = chunk_grid.merge(
            hospitalization[["hospitalization_id", "admission_dttm"]],
            on="hospitalization_id",
            how="left",
        )
        chunk_grid["hours_since_admission"] = (
            chunk_grid["hour_dttm"] - chunk_grid["admission_dttm"]
        ).dt.total_seconds() / 3600.0
        chunk_grid = chunk_grid[chunk_grid["hours_since_admission"] >= min_hours_from_admission].drop(
            columns=["admission_dttm"]
        )
        if progress_callback is not None:
            progress_callback(hi, total)
        yield chunk_grid


def _death_in_window(
    grid: pd.DataFrame,
    hospitalization: pd.DataFrame,
    patient: pd.DataFrame,
) -> pd.Series:
    """True if patient death_dttm falls in (hour_dttm, hour_dttm + 12h] for that row."""
    _ensure_utc(patient, "death_dttm")
    # Grid already has patient_id; merge directly to avoid duplicate column names
    merge = grid.merge(
        patient[["patient_id", "death_dttm"]],
        on="patient_id",
        how="left",
    )
    window_end = merge["hour_dttm"] + pd.Timedelta(hours=12)
    return (
        merge["death_dttm"].notna()
        & (merge["death_dttm"] > merge["hour_dttm"])
        & (merge["death_dttm"] <= window_end)
    ).values


def _event_in_forward_window(
    grid: pd.DataFrame,
    events: pd.DataFrame,
    event_dttm_col: str,
    hid_col: str = "hospitalization_id",
) -> np.ndarray:
    """Vectorized: for each grid row, True if any event in (hour_dttm, hour_dttm+12h]."""
    if events.empty:
        return np.zeros(len(grid), dtype=bool)
    grid = grid.reset_index(drop=True)
    g = grid[[hid_col, "hour_dttm"]].copy()
    g["_gidx"] = np.arange(len(g))
    g["_t1"] = g["hour_dttm"] + pd.Timedelta(hours=12)
    ev = events[[hid_col, event_dttm_col]].drop_duplicates()
    m = g.merge(ev, on=hid_col, how="left")
    if m.empty or m[event_dttm_col].isna().all():
        return np.zeros(len(grid), dtype=bool)
    m["_in_window"] = (m[event_dttm_col] > m["hour_dttm"]) & (m[event_dttm_col] <= m["_t1"])
    out = m.groupby("_gidx")["_in_window"].any()
    return out.reindex(np.arange(len(grid)), fill_value=False).values.astype(bool)


def _vasopressor_start_in_window(
    grid: pd.DataFrame,
    med_continuous: pd.DataFrame,
) -> pd.Series:
    """True if a new vasopressor infusion (dose > 0) starts in (t, t+12h]."""
    _ensure_utc(med_continuous, "admin_dttm")
    mask = med_continuous["med_group"].astype(str).str.lower() == "vasoactives"
    if "mar_action_group" in med_continuous.columns:
        mask = mask & (
            med_continuous["mar_action_group"].astype(str).str.lower() == "administered"
        )
    vaso = med_continuous[mask].copy()
    if vaso.empty:
        return pd.Series(np.zeros(len(grid), dtype=bool))
    vaso = vaso[vaso["med_dose"].fillna(0) > 0][["hospitalization_id", "admin_dttm"]]
    return pd.Series(
        _event_in_forward_window(grid, vaso, "admin_dttm"),
        index=grid.index,
    )


def _imv_start_in_window(
    grid: pd.DataFrame,
    respiratory: pd.DataFrame,
) -> pd.Series:
    """True if IMV (device_category == 'IMV') first appears in (t, t+12h]."""
    _ensure_utc(respiratory, "recorded_dttm")
    imv = respiratory[
        respiratory["device_category"].astype(str).str.upper() == "IMV"
    ][["hospitalization_id", "recorded_dttm"]]
    return pd.Series(
        _event_in_forward_window(grid, imv, "recorded_dttm"),
        index=grid.index,
    )


def _crrt_start_in_window(
    grid: pd.DataFrame,
    crrt: pd.DataFrame,
) -> pd.Series:
    """True if CRRT recorded in (t, t+12h]."""
    _ensure_utc(crrt, "recorded_dttm")
    return pd.Series(
        _event_in_forward_window(
            grid,
            crrt[["hospitalization_id", "recorded_dttm"]],
            "recorded_dttm",
        ),
        index=grid.index,
    )


def _ecmo_start_in_window(
    grid: pd.DataFrame,
    ecmo: pd.DataFrame,
) -> pd.Series:
    """True if ECMO/MCS recorded in (t, t+12h]."""
    _ensure_utc(ecmo, "recorded_dttm")
    return pd.Series(
        _event_in_forward_window(
            grid,
            ecmo[["hospitalization_id", "recorded_dttm"]],
            "recorded_dttm",
        ),
        index=grid.index,
    )


def add_deterioration_label(
    grid: pd.DataFrame,
    hospitalization: pd.DataFrame,
    patient: pd.DataFrame,
    medication_admin_continuous: pd.DataFrame,
    respiratory_support: pd.DataFrame,
    crrt_therapy: pd.DataFrame,
    ecmo_mcs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add binary column 'label': 1 if any of the following in next 12h:
    death, new vasopressor start, new IMV start, CRRT initiation, ECMO initiation.
    """
    out = grid.copy()
    death = _death_in_window(out, hospitalization, patient)
    vaso = _vasopressor_start_in_window(out, medication_admin_continuous)
    imv = _imv_start_in_window(out, respiratory_support)
    crrt = _crrt_start_in_window(out, crrt_therapy)
    ecmo = _ecmo_start_in_window(out, ecmo_mcs)
    out["label"] = (
        pd.Series(death, index=out.index)
        | vaso
        | imv
        | crrt
        | ecmo
    ).astype(int)
    return out


# Threshold above which we label in chunks to avoid OOM (never hold full grid in memory)
_LABEL_CHUNK_THRESHOLD = 80_000


def create_labeled_grid(
    hospitalization: pd.DataFrame,
    patient: pd.DataFrame,
    medication_admin_continuous: pd.DataFrame,
    respiratory_support: pd.DataFrame,
    crrt_therapy: pd.DataFrame,
    ecmo_mcs: pd.DataFrame,
    min_hours_from_admission: int = 6,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """
    Build hourly grid and add deterioration labels in one step.
    For large hospitalization counts (> _LABEL_CHUNK_THRESHOLD), grid is built and labeled
    in chunks to reduce peak memory and avoid OOM kills.
    """
    hosp = hospitalization[["hospitalization_id", "patient_id", "admission_dttm", "discharge_dttm"]].copy()
    _ensure_utc(hosp, "admission_dttm", "discharge_dttm")
    valid = (
        hosp["admission_dttm"].notna()
        & hosp["discharge_dttm"].notna()
        & (hosp["discharge_dttm"] > hosp["admission_dttm"])
    )
    n_hosp = valid.sum()
    if n_hosp == 0:
        return pd.DataFrame(columns=["hospitalization_id", "patient_id", "hour_dttm", "label"])

    if n_hosp > _LABEL_CHUNK_THRESHOLD:
        # Chunked path: yield grid chunks, label each, concat. Never hold full grid.
        labeled_chunks = []
        for grid_chunk in iter_hourly_grid_chunks(
            hospitalization,
            min_hours_from_admission=min_hours_from_admission,
            chunk_size=50_000,
            progress_callback=progress_callback,
        ):
            labeled_chunks.append(
                add_deterioration_label(
                    grid_chunk,
                    hosp,
                    patient,
                    medication_admin_continuous,
                    respiratory_support,
                    crrt_therapy,
                    ecmo_mcs,
                )
            )
        return pd.concat(labeled_chunks, ignore_index=True) if labeled_chunks else pd.DataFrame(columns=["hospitalization_id", "patient_id", "hour_dttm", "label"])
    # Small dataset: build full grid once, then label
    grid = build_hourly_grid(hosp, min_hours_from_admission=min_hours_from_admission, progress_callback=progress_callback)
    if grid.empty:
        grid["label"] = pd.Series(dtype=int)
        return grid
    return add_deterioration_label(
        grid,
        hosp,
        patient,
        medication_admin_continuous,
        respiratory_support,
        crrt_therapy,
        ecmo_mcs,
    )
