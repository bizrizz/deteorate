#!/usr/bin/env python3
"""
Smoke test for SOFA-2 (or SOFA) scoring via clifpy.

1) Loads CLIF config (YAML) to get data_directory, filetype, timezone.
2) Loads hospitalization table, selects 1–3 hospitalizations.
3) Builds a small non-overlapping hourly cohort for those stays.
4) Calls the SOFA calculator (calculate_sofa2 if available, else compute_sofa_polars).
5) Prints hospitalization_id, start_dttm, end_dttm, sofa_total.

Usage:
  python scripts/smoke_test_sofa2.py [--config path/to/clif_config.yaml]
  Or set CLIF_CONFIG_PATH.
"""
import argparse
import os
import sys
from pathlib import Path

# Project root for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Smoke test SOFA scoring with clifpy")
    parser.add_argument(
        "--config",
        default=os.environ.get("CLIF_CONFIG_PATH", str(ROOT / "configs" / "clif_config.yaml")),
        help="Path to CLIF config YAML (data_directory, filetype, timezone)",
    )
    parser.add_argument("--max-hosp", type=int, default=3, help="Max hospitalizations to include in cohort")
    parser.add_argument("--max-hours", type=int, default=24, help="Max hourly windows per hospitalization (0 = no limit)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Load config (clifpy schema: data_directory, filetype, timezone)
    try:
        from clifpy import load_config
        config = load_config(str(config_path))
    except ImportError:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        if "tables_path" in config and "data_directory" not in config:
            config["data_directory"] = config.pop("tables_path")
        for key in ("data_directory", "filetype", "timezone"):
            if key not in config:
                print(f"Config missing required key: {key}", file=sys.stderr)
                sys.exit(1)

    data_directory = config["data_directory"]
    filetype = config.get("filetype", "parquet")
    timezone = config.get("timezone", "UTC")
    if not Path(data_directory).exists():
        print(f"Data directory does not exist: {data_directory}", file=sys.stderr)
        sys.exit(1)

    # Load hospitalization and build small cohort
    from labeling_sofa2 import build_hourly_cohort
    import pandas as pd

    try:
        from data_extraction import load_table
        hospitalization = load_table(data_directory, "hospitalization")
        vitals = load_table(data_directory, "vitals")
    except Exception as e:
        print(f"Failed to load tables: {e}", file=sys.stderr)
        sys.exit(1)

    # Use only hospitalizations that have vitals (so SOFA gets spo2, map, etc.)
    hids_with_vitals = vitals["hospitalization_id"].drop_duplicates()
    hospitalization = hospitalization[
        hospitalization["hospitalization_id"].isin(hids_with_vitals)
    ].drop_duplicates(subset=["hospitalization_id"]).head(args.max_hosp)
    if hospitalization.empty:
        print("No hospitalizations found.", file=sys.stderr)
        sys.exit(1)

    cohort_df = build_hourly_cohort(hospitalization)
    if args.max_hours > 0:
        # Keep first N hours per hospitalization (groupby().apply() drops the key in some pandas versions)
        idx = cohort_df.groupby("hospitalization_id").cumcount() < args.max_hours
        cohort_df = cohort_df.loc[idx].reset_index(drop=True)
    if cohort_df.empty:
        print("Cohort is empty (no valid hourly windows).", file=sys.stderr)
        sys.exit(1)

    print(f"Cohort: {len(cohort_df)} windows, {cohort_df['hospitalization_id'].nunique()} hospitalizations")
    print("Calling SOFA calculator...")

    # Prefer calculate_sofa2 (SOFA-2) if available; else compute_sofa_polars
    out = None
    try:
        from clifpy import calculate_sofa2, SOFA2Config
        sofa_df = calculate_sofa2(
            cohort_df=cohort_df,
            clif_config_path=str(config_path),
            return_rel=False,
            sofa2_config=SOFA2Config(),
        )
        if sofa_df is not None and not (hasattr(sofa_df, "empty") and sofa_df.empty):
            out = sofa_df.to_pandas() if hasattr(sofa_df, "to_pandas") else sofa_df
    except ImportError:
        pass

    if out is None:
        # Use compute_sofa_polars (public clifpy API)
        try:
            import polars as pl
            from clifpy import compute_sofa_polars
        except ImportError as e:
            print(f"clifpy or polars not available: {e}", file=sys.stderr)
            sys.exit(1)
        cohort_pl = pl.from_pandas(
            cohort_df[["hospitalization_id", "start_dttm", "end_dttm"]],
            schema_overrides={
                "start_dttm": pl.Datetime("us", "UTC"),
                "end_dttm": pl.Datetime("us", "UTC"),
            },
        )
        try:
            sofa_df = compute_sofa_polars(
                data_directory=data_directory,
                cohort_df=cohort_pl,
                filetype=filetype,
                timezone=timezone,
            )
            out = sofa_df.to_pandas() if hasattr(sofa_df, "to_pandas") else sofa_df
        except Exception as e:
            print(f"compute_sofa_polars failed: {e}", file=sys.stderr)
            print("\nCohort that was passed to SOFA (hospitalization_id | start_dttm | end_dttm):", file=sys.stderr)
            for _, row in cohort_df.iterrows():
                print(row["hospitalization_id"], row["start_dttm"], row["end_dttm"], file=sys.stderr)
            sys.exit(1)

    # Ensure pandas and required columns
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    if "sofa_total" not in out.columns:
        cand = [c for c in out.columns if "sofa" in c.lower() and "total" in c.lower()]
        if cand:
            out = out.rename(columns={cand[0]: "sofa_total"})
    if "hospitalization_id" not in out.columns or "sofa_total" not in out.columns:
        print(f"Missing required columns. Got: {list(out.columns)}", file=sys.stderr)
        sys.exit(1)
    # compute_sofa_polars returns one row per hospitalization_id; merge with cohort for per-window output
    if "start_dttm" not in out.columns and "end_dttm" not in out.columns:
        out = cohort_df[["hospitalization_id", "start_dttm", "end_dttm"]].merge(
            out[["hospitalization_id", "sofa_total"]].drop_duplicates(),
            on="hospitalization_id",
            how="left",
        )

    print("\nhospitalization_id | start_dttm | end_dttm | sofa_total")
    print("-" * 72)
    for _, row in out.iterrows():
        print(
            str(row["hospitalization_id"]),
            str(row["start_dttm"]),
            str(row["end_dttm"]),
            row["sofa_total"],
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
