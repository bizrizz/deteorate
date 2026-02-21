#!/usr/bin/env python3
"""
Run the full deterioration prediction pipeline.
Usage:
  python run_pipeline.py [--data-dir PATH] [--out-dir PATH]
  or set CLIF_DATA_DIR and run: python run_pipeline.py
"""
import argparse
from pathlib import Path

from train_model import run_pipeline, DEFAULT_DATA_DIR, DEFAULT_OUTPUT_FULL_RUN


def main():
    p = argparse.ArgumentParser(description="ICU deterioration prediction pipeline (full dataset, event-based by default)")
    p.add_argument(
        "--data-dir",
        default=None,
        
        help=f"CLIF parquet data directory (default: CLIF_DATA_DIR or {DEFAULT_DATA_DIR})",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        type=Path,
        help=f"Output directory for model and artifacts (default: {DEFAULT_OUTPUT_FULL_RUN})",
    )
    p.add_argument("--test-size", type=float, default=0.2, help="Fraction for test set (by patient)")
    p.add_argument("--shap-sample", type=int, default=1000, help="Max samples for SHAP (None = all)")
    p.add_argument("--max-windows", type=int, default=None, metavar="N", help="Cap hourly windows (stratified sample) to reduce RAM; e.g. 2000000")
    p.add_argument("--max-rows-per-table", type=int, default=None, metavar="N", help="Cap rows loaded per parquet table to avoid OOM; e.g. 5000000")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sofa2", action="store_true", help="Use SOFA-2 Δ≥2 deterioration label (requires clifpy and --clif-config)")
    p.add_argument("--clif-config", default=None, help="Path to CLIF config for clifpy (or set CLIF_CONFIG_PATH)")
    p.add_argument("--sofa2-delta", type=int, default=2, help="SOFA-2 deterioration threshold (default 2)")
    args = p.parse_args()

    metrics = run_pipeline(
        data_dir=args.data_dir,
        out_dir=args.out_dir or DEFAULT_OUTPUT_FULL_RUN,
        test_size=args.test_size,
        random_state=args.seed,
        shap_sample=args.shap_sample if args.shap_sample is not None else 10000,
        use_sofa2_label=args.sofa2,
        clif_config_path=args.clif_config,
        sofa2_delta_threshold=args.sofa2_delta,
        max_windows=args.max_windows,
        max_rows_per_table=args.max_rows_per_table,
    )
    print("Done. Metrics:", metrics)


if __name__ == "__main__":
    main()
