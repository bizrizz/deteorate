"""
Feature engineering: for each hourly grid row, compute features from the 6-hour lookback window.
Vitals/labs: last, min, max, mean, std, slope, missing indicator.
Medications: vasopressor active (binary), max vasopressor dose.
Respiratory: FiO2 last, PEEP last, mode_category one-hot.
Intake/output: urine output last 6h, net fluid balance.
"""
from typing import Callable, Optional

import numpy as np
import pandas as pd


def _ensure_utc(df: pd.DataFrame, *cols: str) -> None:
    for c in cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            if df[c].dt.tz is None:
                df[c] = df[c].dt.tz_localize("UTC", ambiguous="infer")
        else:
            df[c] = pd.to_datetime(df[c], utc=True)


def _numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


_PROGRESS_REPORT_EVERY = 50_000
_VECTORIZED_CHUNK_HOSP = 25_000  # process this many hospitalizations per chunk in vectorized path
_USE_VECTORIZED_ABOVE_ROWS = 100_000  # use vectorized path when grid has more than this many rows


def _agg_numeric_in_window_vectorized(
    grid: pd.DataFrame,
    subset: pd.DataFrame,
    time_col: str,
    safe_name: str,
    lookback_hours: float = 6.0,
) -> pd.DataFrame:
    """
    Vectorized path: merge grid with data on hospitalization_id (in chunks), filter by time, groupby agg.
    subset must have columns hospitalization_id, time_col, _val (numeric).
    """
    lookback_td = pd.Timedelta(hours=lookback_hours)
    subset = subset.sort_values(time_col)
    unique_hids = grid["hospitalization_id"].unique()
    chunks = [
        unique_hids[i : i + _VECTORIZED_CHUNK_HOSP]
        for i in range(0, len(unique_hids), _VECTORIZED_CHUNK_HOSP)
    ]
    results = []
    for hids in chunks:
        g_chunk = grid[grid["hospitalization_id"].isin(hids)][["hospitalization_id", "hour_dttm"]].drop_duplicates()
        d_chunk = subset[subset["hospitalization_id"].isin(hids)].copy()
        if d_chunk.empty:
            out_chunk = g_chunk.copy()
            out_chunk[safe_name + "_last"] = np.nan
            out_chunk[safe_name + "_min"] = np.nan
            out_chunk[safe_name + "_max"] = np.nan
            out_chunk[safe_name + "_mean"] = np.nan
            out_chunk[safe_name + "_std"] = np.nan
            out_chunk[safe_name + "_slope"] = np.nan
            out_chunk[safe_name + "_missing"] = 1
            results.append(out_chunk)
            continue
        m = g_chunk.merge(
            d_chunk[["hospitalization_id", time_col, "_val"]],
            on="hospitalization_id",
            how="left",
        )
        m = m[(m[time_col] > m["hour_dttm"] - lookback_td) & (m[time_col] <= m["hour_dttm"])]
        if m.empty:
            out_chunk = g_chunk.copy()
            for suf in ["_last", "_min", "_max", "_mean", "_std", "_slope"]:
                out_chunk[safe_name + suf] = np.nan
            out_chunk[safe_name + "_missing"] = 1
            results.append(out_chunk)
            continue
        m["_t"] = (m[time_col] - (m["hour_dttm"] - lookback_td)).dt.total_seconds() / 3600.0
        m = m.sort_values(["_t"])
        agg = m.groupby(["hospitalization_id", "hour_dttm"], as_index=False).agg(
            _last=("_val", "last"),
            _first=("_val", "first"),
            _last_t=("_t", "last"),
            _first_t=("_t", "first"),
            _min=("_val", "min"),
            _max=("_val", "max"),
            _mean=("_val", "mean"),
            _std=("_val", "std"),
        )
        agg[safe_name + "_last"] = agg["_last"]
        agg[safe_name + "_min"] = agg["_min"]
        agg[safe_name + "_max"] = agg["_max"]
        agg[safe_name + "_mean"] = agg["_mean"]
        agg[safe_name + "_std"] = agg["_std"].fillna(0.0)
        dt = agg["_last_t"] - agg["_first_t"]
        agg[safe_name + "_slope"] = np.where(dt > 1e-9, (agg["_last"] - agg["_first"]) / dt, np.nan)
        agg[safe_name + "_missing"] = 0
        out_chunk = g_chunk.merge(
            agg[["hospitalization_id", "hour_dttm", safe_name + "_last", safe_name + "_min", safe_name + "_max", safe_name + "_mean", safe_name + "_std", safe_name + "_slope", safe_name + "_missing"]],
            on=["hospitalization_id", "hour_dttm"],
            how="left",
        )
        out_chunk[safe_name + "_missing"] = out_chunk[safe_name + "_missing"].fillna(1).astype(int)
        for col in [safe_name + "_last", safe_name + "_min", safe_name + "_max", safe_name + "_mean", safe_name + "_std", safe_name + "_slope"]:
            out_chunk[col] = out_chunk[col].astype(float)
        results.append(out_chunk)
    out = pd.concat(results, ignore_index=True)
    full_grid = grid[["hospitalization_id", "hour_dttm"]].drop_duplicates()
    out = full_grid.merge(
        out,
        on=["hospitalization_id", "hour_dttm"],
        how="left",
    )
    for col in [safe_name + "_last", safe_name + "_min", safe_name + "_max", safe_name + "_mean", safe_name + "_std", safe_name + "_slope"]:
        if col in out.columns:
            out[col] = out[col].astype(float)
    out[safe_name + "_missing"] = out[safe_name + "_missing"].fillna(1).astype(int)
    return out


def _agg_numeric_in_window(
    grid: pd.DataFrame,
    data: pd.DataFrame,
    time_col: str,
    value_col: str,
    cat_col: str,
    cat_value: str,
    lookback_hours: float = 6.0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """
    For each (hospitalization_id, hour_dttm), filter data to (hour_dttm - lookback, hour_dttm],
    keep rows where cat_col == cat_value, then compute last, min, max, mean, std, slope, missing.
    If progress_callback(completed, total) is given, it is called every _PROGRESS_REPORT_EVERY hospitalizations.
    """
    _ensure_utc(data, time_col)
    subset = data[data[cat_col].astype(str).str.lower() == cat_value.lower()].copy()
    if subset.empty:
        out = grid[["hospitalization_id", "hour_dttm"]].copy()
        out[f"{cat_value}_last"] = np.nan
        out[f"{cat_value}_min"] = np.nan
        out[f"{cat_value}_max"] = np.nan
        out[f"{cat_value}_mean"] = np.nan
        out[f"{cat_value}_std"] = np.nan
        out[f"{cat_value}_slope"] = np.nan
        out[f"{cat_value}_missing"] = 1
        return out
    subset["_val"] = _numeric_series(subset[value_col])
    subset = subset[["hospitalization_id", time_col, "_val"]].dropna(subset=["_val"])
    if subset.empty:
        out = grid[["hospitalization_id", "hour_dttm"]].copy()
        for suf in ["last", "min", "max", "mean", "std", "slope"]:
            out[f"{cat_value}_{suf}"] = np.nan
        out[f"{cat_value}_missing"] = 1
        return out

    # Normalize column name for vitals/labs (e.g. heart_rate, sodium)
    safe_name = str(cat_value).replace(" ", "_").lower()
    # Pre-group data by hospitalization_id so each grid row only filters a small subset (not full table)
    subset_by_hid = dict(list(subset.groupby("hospitalization_id", sort=False)))
    lookback_td = pd.Timedelta(hours=lookback_hours)
    # Group grid by hospitalization_id only (~400k groups) to avoid 2M-group groupby cost
    n_hosp = grid["hospitalization_id"].nunique()
    results = []
    done = 0
    for hid, grid_grp in grid.groupby("hospitalization_id", sort=False):
        if progress_callback is not None and n_hosp > 0:
            done += 1
            if done % _PROGRESS_REPORT_EVERY == 0 or done == n_hosp:
                progress_callback(done, n_hosp)
        sub = subset_by_hid.get(hid)
        for row in grid_grp.itertuples(index=False):
            hour_dttm = row.hour_dttm
            t_end = hour_dttm
            t_start = t_end - lookback_td
            if sub is None or sub.empty:
                results.append({
                    "hospitalization_id": hid,
                    "hour_dttm": hour_dttm,
                    f"{safe_name}_last": np.nan,
                    f"{safe_name}_min": np.nan,
                    f"{safe_name}_max": np.nan,
                    f"{safe_name}_mean": np.nan,
                    f"{safe_name}_std": np.nan,
                    f"{safe_name}_slope": np.nan,
                    f"{safe_name}_missing": 1,
                })
                continue
            w = sub[(sub[time_col] > t_start) & (sub[time_col] <= t_end)]
            if w.empty:
                results.append({
                    "hospitalization_id": hid,
                    "hour_dttm": hour_dttm,
                    f"{safe_name}_last": np.nan,
                    f"{safe_name}_min": np.nan,
                    f"{safe_name}_max": np.nan,
                    f"{safe_name}_mean": np.nan,
                    f"{safe_name}_std": np.nan,
                    f"{safe_name}_slope": np.nan,
                    f"{safe_name}_missing": 1,
                })
                continue
            w = w.sort_values(time_col)
            vals = w["_val"].values
            times = (w[time_col] - t_start).dt.total_seconds().values / 3600.0
            n = len(vals)
            slope = np.nan
            if n >= 2 and np.ptp(times) > 0:
                slope = float(np.polyfit(times, vals, 1)[0])
            results.append({
                "hospitalization_id": hid,
                "hour_dttm": hour_dttm,
                f"{safe_name}_last": float(vals[-1]),
                f"{safe_name}_min": float(np.nanmin(vals)),
                f"{safe_name}_max": float(np.nanmax(vals)),
                f"{safe_name}_mean": float(np.nanmean(vals)),
                f"{safe_name}_std": float(np.nanstd(vals)) if n > 1 else 0.0,
                f"{safe_name}_slope": slope,
                f"{safe_name}_missing": 0,
            })
    return pd.DataFrame(results)


def build_vital_features(
    grid: pd.DataFrame,
    vitals: pd.DataFrame,
    lookback_hours: float = 6.0,
    vital_categories: Optional[list[str]] = None,
    step_callback: Optional[Callable[[], None]] = None,
    sub_progress_callback: Optional[Callable[[float, int], None]] = None,
) -> pd.DataFrame:
    """
    For each vital_category, compute last, min, max, mean, std, slope, missing in lookback window.
    step_callback is called after each vital category. sub_progress_callback(global_current, total) is
    called during each category (every N hospitalizations) so the bar moves; global_current is fractional (e.g. 0.5 = halfway through first vital).
    """
    if vital_categories is None:
        vital_categories = [
            "heart_rate",
            "sbp",
            "dbp",
            "map",
            "spo2",
            "respiratory_rate",
            "temp_c",
        ]
    _ensure_utc(vitals, "recorded_dttm")
    out = grid[["hospitalization_id", "hour_dttm"]].drop_duplicates()
    use_vectorized = len(grid) > _USE_VECTORIZED_ABOVE_ROWS
    for i, vcat in enumerate(vital_categories):
        safe_name = str(vcat).replace(" ", "_").lower()
        if use_vectorized:
            subset = vitals[vitals["vital_category"].astype(str).str.lower() == vcat.lower()].copy()
            subset["_val"] = _numeric_series(subset["vital_value"])
            subset = subset[["hospitalization_id", "recorded_dttm", "_val"]].dropna(subset=["_val"])
            if subset.empty:
                out[safe_name + "_last"] = np.nan
                out[safe_name + "_min"] = np.nan
                out[safe_name + "_max"] = np.nan
                out[safe_name + "_mean"] = np.nan
                out[safe_name + "_std"] = np.nan
                out[safe_name + "_slope"] = np.nan
                out[safe_name + "_missing"] = 1
            else:
                agg = _agg_numeric_in_window_vectorized(
                    grid, subset, "recorded_dttm", safe_name, lookback_hours
                )
                out = out.merge(agg, on=["hospitalization_id", "hour_dttm"], how="left")
        else:
            def _cb(c: int, t: int, _i: int = i) -> None:
                if sub_progress_callback is not None and t > 0:
                    sub_progress_callback(_i + c / t, _FEATURES_TOTAL_STEPS)
            agg = _agg_numeric_in_window(
                grid, vitals, "recorded_dttm", "vital_value", "vital_category", vcat,
                lookback_hours, progress_callback=_cb if sub_progress_callback else None,
            )
            out = out.merge(agg, on=["hospitalization_id", "hour_dttm"], how="left")
        if sub_progress_callback is not None:
            sub_progress_callback(i + 1, _FEATURES_TOTAL_STEPS)
        if step_callback is not None:
            step_callback()
    return out


def build_lab_features(
    grid: pd.DataFrame,
    labs: pd.DataFrame,
    lookback_hours: float = 6.0,
    lab_categories: Optional[list[str]] = None,
    step_callback: Optional[Callable[[], None]] = None,
    sub_progress_callback: Optional[Callable[[float, int], None]] = None,
) -> pd.DataFrame:
    """
    For each lab_category, compute last, min, max, mean, std, slope, missing in lookback window.
    sub_progress_callback(global_current_float, total) is called during each category for progress.
    """
    if lab_categories is None:
        lab_categories = [
            "sodium",
            "potassium",
            "chloride",
            "glucose",
            "creatinine",
            "bun",
            "hemoglobin",
            "white_blood_cell_count",
            "platelet_count",
            "lactate",
            "ph",
            "bicarbonate",
        ]
    labs = labs.copy()
    _ensure_utc(labs, "lab_result_dttm")
    if "lab_value_numeric" in labs.columns:
        labs["_lab_val"] = _numeric_series(labs["lab_value_numeric"])
    else:
        labs["_lab_val"] = _numeric_series(labs["lab_value"])
    labs = labs.rename(columns={"lab_result_dttm": "recorded_dttm", "_lab_val": "vital_value"})
    labs["vital_category"] = labs["lab_category"]
    out = grid[["hospitalization_id", "hour_dttm"]].drop_duplicates()
    use_vectorized = len(grid) > _USE_VECTORIZED_ABOVE_ROWS
    for i, lcat in enumerate(lab_categories):
        subset = labs[labs["lab_category"].astype(str).str.lower() == lcat.lower()]
        if subset.empty:
            continue
        safe_name = str(lcat).replace(" ", "_").lower()
        if use_vectorized:
            sub = subset[["hospitalization_id", "recorded_dttm", "vital_value"]].copy()
            sub["_val"] = _numeric_series(sub["vital_value"])
            sub = sub[["hospitalization_id", "recorded_dttm", "_val"]].dropna(subset=["_val"])
            if sub.empty:
                out[safe_name + "_last"] = np.nan
                out[safe_name + "_min"] = np.nan
                out[safe_name + "_max"] = np.nan
                out[safe_name + "_mean"] = np.nan
                out[safe_name + "_std"] = np.nan
                out[safe_name + "_slope"] = np.nan
                out[safe_name + "_missing"] = 1
            else:
                agg = _agg_numeric_in_window_vectorized(
                    grid, sub, "recorded_dttm", safe_name, lookback_hours
                )
                out = out.merge(agg, on=["hospitalization_id", "hour_dttm"], how="left")
        else:
            def _cb(c: int, t: int, _i: int = i) -> None:
                if sub_progress_callback is not None and t > 0:
                    sub_progress_callback(_N_VITAL_CATS + _i + c / t, _FEATURES_TOTAL_STEPS)
            agg = _agg_numeric_in_window(
                grid, subset, "recorded_dttm", "vital_value", "lab_category", lcat,
                lookback_hours, progress_callback=_cb if sub_progress_callback else None,
            )
            out = out.merge(agg, on=["hospitalization_id", "hour_dttm"], how="left")
        if sub_progress_callback is not None:
            sub_progress_callback(_N_VITAL_CATS + i + 1, _FEATURES_TOTAL_STEPS)
        if step_callback is not None:
            step_callback()
    return out


def build_medication_features(
    grid: pd.DataFrame,
    med_continuous: pd.DataFrame,
    lookback_hours: float = 6.0,
) -> pd.DataFrame:
    """Vasopressor active (binary), max vasopressor dose in window."""
    _ensure_utc(med_continuous, "admin_dttm")
    vaso = med_continuous[
        med_continuous["med_group"].astype(str).str.lower() == "vasoactives"
    ].copy()
    vaso["_dose"] = _numeric_series(vaso["med_dose"]).fillna(0)
    if "mar_action_group" in vaso.columns:
        vaso = vaso[vaso["mar_action_group"].astype(str).str.lower() == "administered"]
    vaso = vaso[["hospitalization_id", "admin_dttm", "_dose"]]
    lookback_td = pd.Timedelta(hours=lookback_hours)
    if len(grid) > _USE_VECTORIZED_ABOVE_ROWS:
        # Vectorized: chunked merge + filter + groupby
        unique_hids = grid["hospitalization_id"].unique()
        chunks = [unique_hids[i : i + _VECTORIZED_CHUNK_HOSP] for i in range(0, len(unique_hids), _VECTORIZED_CHUNK_HOSP)]
        results = []
        for hids in chunks:
            g_chunk = grid[grid["hospitalization_id"].isin(hids)][["hospitalization_id", "hour_dttm"]].drop_duplicates()
            v_chunk = vaso[vaso["hospitalization_id"].isin(hids)]
            if v_chunk.empty:
                g_chunk["vasopressor_active"] = 0
                g_chunk["vasopressor_max_dose"] = 0.0
                results.append(g_chunk)
                continue
            m = g_chunk.merge(v_chunk, on="hospitalization_id", how="left")
            m = m[(m["admin_dttm"] > m["hour_dttm"] - lookback_td) & (m["admin_dttm"] <= m["hour_dttm"]) & (m["_dose"] > 0)]
            if m.empty:
                g_chunk["vasopressor_active"] = 0
                g_chunk["vasopressor_max_dose"] = 0.0
                results.append(g_chunk)
                continue
            agg = m.groupby(["hospitalization_id", "hour_dttm"], as_index=False).agg(
                vasopressor_max_dose=("_dose", "max"),
            )
            agg["vasopressor_active"] = 1
            out_chunk = g_chunk.merge(agg, on=["hospitalization_id", "hour_dttm"], how="left")
            out_chunk["vasopressor_active"] = out_chunk["vasopressor_active"].fillna(0).astype(int)
            out_chunk["vasopressor_max_dose"] = out_chunk["vasopressor_max_dose"].fillna(0.0)
            results.append(out_chunk)
        out = pd.concat(results, ignore_index=True)
        full = grid[["hospitalization_id", "hour_dttm"]].drop_duplicates()
        return full.merge(out, on=["hospitalization_id", "hour_dttm"], how="left").fillna({"vasopressor_active": 0, "vasopressor_max_dose": 0.0})
    vaso_by_hid = dict(list(vaso.groupby("hospitalization_id", sort=False)))
    empty_vaso = vaso.iloc[:0]
    results = []
    for hid, grid_grp in grid.groupby("hospitalization_id", sort=False):
        v = vaso_by_hid.get(hid, empty_vaso)
        for row in grid_grp.itertuples(index=False):
            hour_dttm = row.hour_dttm
            t_end = hour_dttm
            t_start = t_end - lookback_td
            w = v[(v["admin_dttm"] > t_start) & (v["admin_dttm"] <= t_end) & (v["_dose"] > 0)]
            results.append({
                "hospitalization_id": hid,
                "hour_dttm": hour_dttm,
                "vasopressor_active": 1 if len(w) > 0 else 0,
                "vasopressor_max_dose": float(w["_dose"].max()) if len(w) > 0 else 0.0,
            })
    return pd.DataFrame(results)


def build_respiratory_features(
    grid: pd.DataFrame,
    respiratory: pd.DataFrame,
    lookback_hours: float = 6.0,
) -> pd.DataFrame:
    """FiO2 last, PEEP last, mode_category one-hot (last mode in window)."""
    _ensure_utc(respiratory, "recorded_dttm")
    resp = respiratory[
        ["hospitalization_id", "recorded_dttm", "fio2_set", "peep_set", "mode_category"]
    ].copy()
    resp["fio2_set"] = _numeric_series(resp["fio2_set"])
    resp["peep_set"] = _numeric_series(resp["peep_set"])
    lookback_td = pd.Timedelta(hours=lookback_hours)
    if len(grid) > _USE_VECTORIZED_ABOVE_ROWS:
        unique_hids = grid["hospitalization_id"].unique()
        chunks = [unique_hids[i : i + _VECTORIZED_CHUNK_HOSP] for i in range(0, len(unique_hids), _VECTORIZED_CHUNK_HOSP)]
        results = []
        for hids in chunks:
            g_chunk = grid[grid["hospitalization_id"].isin(hids)][["hospitalization_id", "hour_dttm"]].drop_duplicates()
            r_chunk = resp[resp["hospitalization_id"].isin(hids)]
            if r_chunk.empty:
                g_chunk["fio2_last"] = np.nan
                g_chunk["peep_last"] = np.nan
                g_chunk["mode_category_last"] = ""
                results.append(g_chunk)
                continue
            m = g_chunk.merge(r_chunk, on="hospitalization_id", how="left")
            m = m[(m["recorded_dttm"] > m["hour_dttm"] - lookback_td) & (m["recorded_dttm"] <= m["hour_dttm"])]
            if m.empty:
                g_chunk["fio2_last"] = np.nan
                g_chunk["peep_last"] = np.nan
                g_chunk["mode_category_last"] = ""
                results.append(g_chunk)
                continue
            m = m.sort_values("recorded_dttm")
            agg = m.groupby(["hospitalization_id", "hour_dttm"], as_index=False).agg(
                fio2_last=("fio2_set", "last"),
                peep_last=("peep_set", "last"),
                mode_category_last=("mode_category", "last"),
            )
            agg["mode_category_last"] = agg["mode_category_last"].astype(str).fillna("")
            out_chunk = g_chunk.merge(agg, on=["hospitalization_id", "hour_dttm"], how="left")
            results.append(out_chunk)
        out = pd.concat(results, ignore_index=True)
        full = grid[["hospitalization_id", "hour_dttm"]].drop_duplicates()
        out = full.merge(out, on=["hospitalization_id", "hour_dttm"], how="left")
    else:
        resp_by_hid = dict(list(resp.groupby("hospitalization_id", sort=False)))
        empty_resp = resp.iloc[:0]
        results = []
        for hid, grid_grp in grid.groupby("hospitalization_id", sort=False):
            r = resp_by_hid.get(hid, empty_resp)
            for row in grid_grp.itertuples(index=False):
                hour_dttm = row.hour_dttm
                t_end = hour_dttm
                t_start = t_end - lookback_td
                w = r[(r["recorded_dttm"] > t_start) & (r["recorded_dttm"] <= t_end)].sort_values("recorded_dttm")
                if w.empty:
                    results.append({"hospitalization_id": hid, "hour_dttm": hour_dttm, "fio2_last": np.nan, "peep_last": np.nan, "mode_category_last": ""})
                else:
                    last = w.iloc[-1]
                    results.append({
                        "hospitalization_id": hid, "hour_dttm": hour_dttm,
                        "fio2_last": last["fio2_set"], "peep_last": last["peep_set"],
                        "mode_category_last": str(last["mode_category"]) if pd.notna(last["mode_category"]) else "",
                    })
        out = pd.DataFrame(results)
    mode_dummies = pd.get_dummies(out["mode_category_last"], prefix="mode")
    out = pd.concat([out.drop(columns=["mode_category_last"]), mode_dummies], axis=1)
    return out


def build_intake_output_features(
    grid: pd.DataFrame,
    intake_output: Optional[pd.DataFrame],
    lookback_hours: float = 6.0,
) -> pd.DataFrame:
    """Urine output last 6h, net fluid balance (intake - output) in window. CLIF: in_out_flag 1=intake, 0=output."""
    lookback_td = pd.Timedelta(hours=lookback_hours)
    if intake_output is None or intake_output.empty:
        out = grid[["hospitalization_id", "hour_dttm"]].drop_duplicates()
        out["urine_output_6h"] = np.nan
        out["net_fluid_balance"] = np.nan
        return out
    time_col = "intake_dttm" if "intake_dttm" in intake_output.columns else None
    if time_col is None:
        for c in intake_output.columns:
            if "dttm" in c.lower():
                time_col = c
                break
    if time_col is None:
        out = grid[["hospitalization_id", "hour_dttm"]].drop_duplicates()
        out["urine_output_6h"] = np.nan
        out["net_fluid_balance"] = np.nan
        return out
    _ensure_utc(intake_output, time_col)
    io = intake_output.copy()
    io["_amount"] = _numeric_series(io["amount"]).fillna(0)
    has_in_out = "in_out_flag" in io.columns
    has_fluid = "fluid_name" in io.columns
    if len(grid) > _USE_VECTORIZED_ABOVE_ROWS:
        unique_hids = grid["hospitalization_id"].unique()
        chunks = [unique_hids[i : i + _VECTORIZED_CHUNK_HOSP] for i in range(0, len(unique_hids), _VECTORIZED_CHUNK_HOSP)]
        results = []
        for hids in chunks:
            g_chunk = grid[grid["hospitalization_id"].isin(hids)][["hospitalization_id", "hour_dttm"]].drop_duplicates()
            io_chunk = io[io["hospitalization_id"].isin(hids)]
            if io_chunk.empty:
                g_chunk["urine_output_6h"] = np.nan
                g_chunk["net_fluid_balance"] = np.nan
                results.append(g_chunk)
                continue
            m = g_chunk.merge(io_chunk, on="hospitalization_id", how="left")
            m = m[(m[time_col] > m["hour_dttm"] - lookback_td) & (m[time_col] <= m["hour_dttm"])]
            if m.empty:
                g_chunk["urine_output_6h"] = np.nan
                g_chunk["net_fluid_balance"] = np.nan
                results.append(g_chunk)
                continue
            if has_in_out:
                m["_intake"] = np.where(m["in_out_flag"] == 1, m["_amount"], 0)
                m["_output"] = np.where(m["in_out_flag"] == 0, m["_amount"], 0)
                m["_net"] = m["_intake"] - m["_output"]
            else:
                m["_net"] = 0
            if has_fluid:
                fluid = m["fluid_name"].astype(str).str.lower()
                urine = fluid.str.contains("urine", na=False)
                m["_urine"] = np.where(urine & (m["in_out_flag"] == 0 if has_in_out else True), m["_amount"], 0)
            else:
                m["_urine"] = 0
            agg = m.groupby(["hospitalization_id", "hour_dttm"], as_index=False).agg(
                urine_output_6h=("_urine", "sum"),
                net_fluid_balance=("_net", "sum"),
            )
            if not has_in_out:
                agg["net_fluid_balance"] = np.nan
            if not has_fluid:
                agg["urine_output_6h"] = np.nan
            out_chunk = g_chunk.merge(agg, on=["hospitalization_id", "hour_dttm"], how="left")
            results.append(out_chunk)
        out = pd.concat(results, ignore_index=True)
        full = grid[["hospitalization_id", "hour_dttm"]].drop_duplicates()
        return full.merge(out, on=["hospitalization_id", "hour_dttm"], how="left")
    io_by_hid = dict(list(io.groupby("hospitalization_id", sort=False)))
    empty_io = io.iloc[:0]
    results = []
    for hid, grid_grp in grid.groupby("hospitalization_id", sort=False):
        w_io = io_by_hid.get(hid, empty_io)
        for row in grid_grp.itertuples(index=False):
            hour_dttm = row.hour_dttm
            t_end = hour_dttm
            t_start = t_end - lookback_td
            urine_output_6h = np.nan
            net_balance = np.nan
            if not w_io.empty:
                w = w_io[(w_io[time_col] > t_start) & (w_io[time_col] <= t_end)]
                if not w.empty:
                    amount = w["_amount"]
                    if has_in_out:
                        in_out = w["in_out_flag"]
                        intake_sum = amount[in_out == 1].sum()
                        output_sum = amount[in_out == 0].sum()
                        net_balance = float(intake_sum - output_sum)
                        if has_fluid:
                            fluid = w["fluid_name"].astype(str).str.lower()
                            urine_mask = fluid.str.contains("urine", na=False) & (in_out == 0)
                            urine_output_6h = float(amount[urine_mask].sum())
                    elif has_fluid:
                        fluid = w["fluid_name"].astype(str).str.lower()
                        urine_output_6h = float(amount[fluid.str.contains("urine", na=False)].sum())
            results.append({
                "hospitalization_id": hid,
                "hour_dttm": hour_dttm,
                "urine_output_6h": urine_output_6h,
                "net_fluid_balance": net_balance,
            })
    return pd.DataFrame(results)


# Number of sub-steps for Features progress (vitals + labs + med + resp + io)
_N_VITAL_CATS = 7
_N_LAB_CATS = 12
_FEATURES_TOTAL_STEPS = _N_VITAL_CATS + _N_LAB_CATS + 3  # +3 for med, resp, io


def build_all_features(
    grid: pd.DataFrame,
    vitals: pd.DataFrame,
    labs: pd.DataFrame,
    medication_admin_continuous: pd.DataFrame,
    respiratory_support: pd.DataFrame,
    intake_output: Optional[pd.DataFrame] = None,
    lookback_hours: float = 6.0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """
    Build all feature blocks and merge on (hospitalization_id, hour_dttm).
    If progress_callback(current, total) is given, it is called after each sub-step (vital, lab, or block)
    so the bar moves during the long vitals/labs work.
    """
    X = grid[["hospitalization_id", "hour_dttm", "patient_id"]].drop_duplicates()
    total_steps = _FEATURES_TOTAL_STEPS
    step = [0]  # mutable so step_callback can increment
    if progress_callback is not None:
        progress_callback(0, total_steps)  # show bar at 0% immediately

    def step_callback() -> None:
        if progress_callback is not None:
            step[0] += 1
            progress_callback(step[0], total_steps)

    # Build one block at a time; sub_progress_callback reports during each vital/lab (every 50k hosps)
    feat_df = build_vital_features(
        grid, vitals, lookback_hours, step_callback=step_callback, sub_progress_callback=progress_callback
    )
    cols = [c for c in feat_df.columns if c not in ["hospitalization_id", "hour_dttm"]]
    if cols:
        X = X.merge(
            feat_df[["hospitalization_id", "hour_dttm"] + cols],
            on=["hospitalization_id", "hour_dttm"],
            how="left",
        )

    feat_df = build_lab_features(
        grid, labs, lookback_hours, step_callback=step_callback, sub_progress_callback=progress_callback
    )
    cols = [c for c in feat_df.columns if c not in ["hospitalization_id", "hour_dttm"]]
    if cols:
        X = X.merge(
            feat_df[["hospitalization_id", "hour_dttm"] + cols],
            on=["hospitalization_id", "hour_dttm"],
            how="left",
        )

    if progress_callback is not None:
        step[0] += 1
        progress_callback(step[0], total_steps)
    feat_df = build_medication_features(grid, medication_admin_continuous, lookback_hours)
    cols = [c for c in feat_df.columns if c not in ["hospitalization_id", "hour_dttm"]]
    if cols:
        X = X.merge(
            feat_df[["hospitalization_id", "hour_dttm"] + cols],
            on=["hospitalization_id", "hour_dttm"],
            how="left",
        )

    if progress_callback is not None:
        step[0] += 1
        progress_callback(step[0], total_steps)
    feat_df = build_respiratory_features(grid, respiratory_support, lookback_hours)
    cols = [c for c in feat_df.columns if c not in ["hospitalization_id", "hour_dttm"]]
    if cols:
        X = X.merge(
            feat_df[["hospitalization_id", "hour_dttm"] + cols],
            on=["hospitalization_id", "hour_dttm"],
            how="left",
        )

    if progress_callback is not None:
        step[0] += 1
        progress_callback(step[0], total_steps)
    feat_df = build_intake_output_features(grid, intake_output, lookback_hours)
    cols = [c for c in feat_df.columns if c not in ["hospitalization_id", "hour_dttm"]]
    if cols:
        X = X.merge(
            feat_df[["hospitalization_id", "hour_dttm"] + cols],
            on=["hospitalization_id", "hour_dttm"],
            how="left",
        )

    if progress_callback is not None:
        progress_callback(total_steps, total_steps)
    return X
