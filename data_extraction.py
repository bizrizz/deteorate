"""
Load CLIF-MIMIC parquet tables for the deterioration prediction pipeline.
Data dictionary: https://clif-icu.com/data-dictionary/data-dictionary-2.1.0
"""
import os
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False


# Default path to CLIF-MIMIC Release 1.0.0 (no closing paren in folder name)
DEFAULT_DATA_DIR = os.environ.get(
    "CLIF_DATA_DIR",
    "/Users/adambizios/Downloads/CLIF-MIMIC/Release 1.0.0 (MIMIC-IV 3.1 - CLIF 2",
)


def load_table(
    data_dir: str,
    name: str,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a single CLIF parquet table by base name (e.g. 'hospitalization').
    If max_rows is set, read in row-groups/chunks and stop after that many rows
    to limit memory (requires pyarrow).
    """
    path = Path(data_dir) / f"clif_{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Table not found: {path}")
    if max_rows is None or not _HAS_PYARROW:
        return pd.read_parquet(path)
    # Chunked read to avoid loading full table into memory
    import pyarrow as pa
    pf = pq.ParquetFile(path)
    batch_size = min(500_000, max_rows)
    batches = []
    rows_so_far = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        batches.append(batch)
        rows_so_far += batch.num_rows
        if rows_so_far >= max_rows:
            break
    if not batches:
        return pd.DataFrame()
    table = pa.Table.from_batches(batches)
    if table.num_rows > max_rows:
        table = table.slice(0, max_rows)
    return table.to_pandas()


def load_all_tables(
    data_dir: Optional[str] = None,
    include_intake_output: bool = True,
    max_rows_per_table: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """
    Load all tables required for the deterioration pipeline.
    Returns dict of table_name -> DataFrame.
    intake_output is optional (Concept table; may be missing in some releases).
    If max_rows_per_table is set, each table is capped at that many rows (chunked read)
    to reduce memory use and avoid OOM on large CLIF datasets.
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    data_dir = str(Path(data_dir).expanduser().resolve())
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Set CLIF_DATA_DIR to your CLIF-MIMIC folder or run with: --data-dir /path/to/CLIF-MIMIC"
        )

    tables = {}

    required = [
        "hospitalization",
        "vitals",
        "labs",
        "respiratory_support",
        "medication_admin_continuous",
        "crrt_therapy",
        "ecmo_mcs",
        "patient",
    ]
    for name in required:
        tables[name] = load_table(data_dir, name, max_rows=max_rows_per_table)

    if include_intake_output:
        try:
            tables["intake_output"] = load_table(data_dir, "intake_output", max_rows=max_rows_per_table)
        except FileNotFoundError:
            tables["intake_output"] = None  # type: ignore

    return tables


def get_hospitalization_bounds(
    hospitalization: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return hospitalization_id, patient_id, admission_dttm, discharge_dttm
    with dttm columns as datetime.
    """
    cols = ["hospitalization_id", "patient_id", "admission_dttm", "discharge_dttm"]
    df = hospitalization[cols].copy()
    for c in ["admission_dttm", "discharge_dttm"]:
        if c in df.columns and pd.api.types.is_object_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], utc=True)
    return df
