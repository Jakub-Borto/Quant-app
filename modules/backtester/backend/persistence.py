"""
Trades persistence for the Backtester module.

Extracted verbatim from legacy_streamlit/views/backtester.py. Saved trades
carry parquet key-value metadata recording the filter state under which they
were saved (read back by Analytics), and a filter-aware dedup prevents saving
byte-identical results twice.

Change vs the old view: the output directory is passed in (`trades_dir`,
usually `<data_root>/trades`) instead of the old hardcoded data/trades.
"""

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _build_filter_metadata(filtered: bool, selected_day_types: list,
                           selected_trade_types) -> dict:
    """
    Build the parquet key-value metadata (bytes->bytes) recording the active
    filter state. selected_trade_types is either the string "all" or a list.
    """
    if selected_trade_types == "all":
        tt_value = b"all"
    else:
        tt_value = json.dumps(list(selected_trade_types)).encode()

    return {
        b"filtered":             b"true" if filtered else b"false",
        b"selected_day_types":   json.dumps(list(selected_day_types)).encode(),
        b"selected_trade_types": tt_value,
    }


def _read_filter_metadata(path: Path) -> dict:
    """Return the filter kv-metadata subset of an existing trades parquet."""
    schema_meta = pq.read_schema(path).metadata or {}
    keys = (b"filtered", b"selected_day_types", b"selected_trade_types")
    return {k: schema_meta.get(k) for k in keys}


def save_trades(trades_dir: Path, trades: pd.DataFrame, dataset: str,
                strategy: str, start_date, end_date, filtered: bool,
                selected_day_types: list, selected_trade_types) -> str | None:
    """
    Write trades to {trades_dir}/{dataset}_{strategy}_{start}_{end}
    [_filtered][_N].parquet with the filter kv-metadata attached.

    Returns the written path as str, or None when an identical file (same row
    content AND same filter metadata) already exists.
    """
    trades_path = Path(trades_dir)
    trades_path.mkdir(parents=True, exist_ok=True)

    base_name = f"{dataset}_{strategy}_{start_date}_{end_date}"
    stem      = base_name + ("_filtered" if filtered else "")

    new_meta = _build_filter_metadata(filtered, selected_day_types, selected_trade_types)

    # Filter-aware dedup: a re-save is a duplicate only when BOTH the row
    # content and the filter metadata match an existing file.
    for f in sorted(trades_path.glob(f"{stem}*.parquet")):
        if pd.read_parquet(f).equals(trades) and _read_filter_metadata(f) == new_meta:
            return None

    output_path = trades_path / f"{stem}.parquet"
    n = 2
    while output_path.exists():
        output_path = trades_path / f"{stem}_{n}.parquet"
        n += 1

    # Write via pyarrow so we can attach kv metadata; from_pandas keeps the
    # b'pandas' schema metadata so pd.read_parquet reconstructs the frame.
    table = pa.Table.from_pandas(trades)
    meta  = dict(table.schema.metadata or {})
    meta.update(new_meta)
    table = table.replace_schema_metadata(meta)
    pq.write_table(table, output_path)

    return str(output_path)
