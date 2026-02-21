# ICU Deterioration Prediction Model

Predicts clinical deterioration within the next 12 hours using a 6-hour lookback window of structured ICU data in [CLIF 2.1.0](https://clif-icu.com/data-dictionary/data-dictionary-2.1.0) format (e.g. CLIF-MIMIC).

## Setup

```bash
cd /Users/adambizios/deteorate
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Data

Point the pipeline at your CLIF parquet directory (e.g. CLIF-MIMIC Release 1.0.0):

```bash
export CLIF_DATA_DIR="/Users/adambizios/Downloads/CLIF-MIMIC/Release 1.0.0 (MIMIC-IV 3.1 - CLIF 2)"
```

Required tables: `hospitalization`, `vitals`, `labs`, `respiratory_support`, `medication_admin_continuous`, `crrt_therapy`, `ecmo_mcs`, `patient`. Optional: `intake_output`.

## Label

Two options (choose in code or CLI):

**Event-based (default)**  
- **Label = 1** if within the **next 12 hours**: death, new vasopressor start, new mechanical ventilation (IMV) start, CRRT initiation, or ECMO initiation.  
- **Label = 0** otherwise.

**SOFA-2–based (optional)**  
- Uses **clifpy** (`calculate_sofa2`, `SOFA2Config`); no re-implementation of SOFA logic.  
- Non-overlapping hourly windows from admission to discharge; `calculate_sofa2` is run once on that cohort.  
- **Baseline**: max SOFA-2 total over the previous 6 hourly windows (t−6h..t].  
- **Future max**: max SOFA-2 total over the next 12 hourly windows (t..t+12h].  
- **Label = 1** if (future_max − baseline_max) ≥ 2 (configurable), else 0.  
- Requires a CLIF config path (for clifpy to load data) and `pip install clifpy`.

## Pipeline

1. **data_extraction.py** – Load CLIF parquet tables.
2. **labeling.py** – Hourly grid; event-based deterioration label.  
   **labeling_sofa2.py** – SOFA-2–based label via clifpy (non-overlapping hourly cohort → `calculate_sofa2` → rolling baseline/future max → Δ ≥ 2).
3. **feature_engineering.py** – 6-hour lookback features:
   - Vitals/labs: last, min, max, mean, std, slope, missing.
   - Medications: vasopressor active, max vasopressor dose.
   - Respiratory: FiO2 last, PEEP last, mode one-hot.
   - Intake/output: urine output 6h, net fluid balance.
4. **train_model.py** – XGBoost, split by `patient_id`, AUROC/AUPRC, save model, SHAP.

## Run

**Event-based label (default):**
```bash
python train_model.py
```

**SOFA-2 label** (set `CLIF_CONFIG_PATH` to your clifpy config file, or pass `--clif-config`):
```bash
export CLIF_CONFIG_PATH=/path/to/your/clif_config.yaml
python run_pipeline.py --sofa2 --clif-config "$CLIF_CONFIG_PATH"
```

Or from Python:

```python
from train_model import run_pipeline

# Event-based label
metrics = run_pipeline(
    data_dir="/path/to/CLIF-MIMIC/Release 1.0.0 (MIMIC-IV 3.1 - CLIF 2)",
    out_dir="./output",
    test_size=0.2,
    shap_sample=1000,
)

# SOFA-2 label (ΔSOFA-2 ≥ 2)
metrics = run_pipeline(
    data_dir="/path/to/CLIF-MIMIC/...",
    out_dir="./output",
    use_sofa2_label=True,
    clif_config_path="/path/to/clif_config.yaml",
    sofa2_delta_threshold=2,
)
```

Outputs in `out_dir`: `model.pkl`, `deterioration_xgb_model.json`, `feature_list.json`, `metrics.json`, `shap_importance.csv`, `predictions.parquet` (window-level predictions for the API).

## API (Lovable-friendly)

A FastAPI service serves the saved predictions and an explain endpoint:

```bash
# Optional: set artifacts dir (default: ./output_full_run) and CLIF data dir (required for /explain)
export DETECTORATE_ARTIFACTS_DIR=./output_full_run
export CLIF_DATA_DIR="/path/to/CLIF-MIMIC/Release 1.0.0 (MIMIC-IV 3.1 - CLIF 2)"

python api.py
# or: uvicorn api:app --host 0.0.0.0 --port 8000
```

- **GET /patients?limit=50&search=** – List hospitalizations with latest timestamp and risk_score; optional `search` filters by hospitalization_id or patient_id; sorted by risk_score descending.
- **GET /explain?hospitalization_id=...&timestamp=...** – Returns risk_score, label, top SHAP drivers, and timeseries (hr, map, spo2, fio2, peep, lactate, creatinine) for the 6h lookback. Requires `CLIF_DATA_DIR` for feature reconstruction and timeseries.
