"""
Saved-trades parquet persistence — shared by the Backtester and the
Optimizer cell detail (the Save Trades / Go to Analytics / Go to Monte Carlo
action row).

Two writers, same filter kv-metadata (read back by Analytics):

- save_trades: regular saves into <data_root>/trades/ under a caller-built
  base name, with a filter-aware dedup (extracted verbatim from
  legacy_streamlit/views/backtester.py, generalized so the base name is the
  caller's: {dataset}_{strategy}_{start}_{end} for the backtester,
  ticker + run + cell params for the optimizer).
- save_temp_trades: one-session handoff files into <data_root>/temp/ as
  {ASSET}_temp_file_{N}.parquet (asset prefix keeps the "asset = first
  underscore token of the filename" convention working downstream; app boot
  clears them via data_roots.clear_temp_files).

Qt-free.
"""

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def build_filter_metadata(filtered: bool, selected_day_types: list,
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


def read_filter_metadata(path: Path) -> dict:
    """Return the filter kv-metadata subset of an existing trades parquet."""
    schema_meta = pq.read_schema(path).metadata or {}
    keys = (b"filtered", b"selected_day_types", b"selected_trade_types")
    return {k: schema_meta.get(k) for k in keys}


def write_with_metadata(path: Path, trades: pd.DataFrame, kv_meta: dict) -> None:
    # Write via pyarrow so we can attach kv metadata; from_pandas keeps the
    # b'pandas' schema metadata so pd.read_parquet reconstructs the frame.
    table = pa.Table.from_pandas(trades)
    meta = dict(table.schema.metadata or {})
    meta.update(kv_meta)
    pq.write_table(table.replace_schema_metadata(meta), path)


def save_trades(trades_dir: Path, trades: pd.DataFrame, base_name: str,
                filtered: bool, selected_day_types: list,
                selected_trade_types) -> str | None:
    """
    Write trades to {trades_dir}/{base_name}[_filtered][_N].parquet with the
    filter kv-metadata attached.

    Returns the written path as str, or None when an identical file (same row
    content AND same filter metadata) already exists.
    """
    trades_path = Path(trades_dir)
    trades_path.mkdir(parents=True, exist_ok=True)

    stem = base_name + ("_filtered" if filtered else "")

    new_meta = build_filter_metadata(filtered, selected_day_types, selected_trade_types)

    # Filter-aware dedup: a re-save is a duplicate only when BOTH the row
    # content and the filter metadata match an existing file.
    for f in sorted(trades_path.glob(f"{stem}*.parquet")):
        if pd.read_parquet(f).equals(trades) and read_filter_metadata(f) == new_meta:
            return None

    output_path = trades_path / f"{stem}.parquet"
    n = 2
    while output_path.exists():
        output_path = trades_path / f"{stem}_{n}.parquet"
        n += 1

    write_with_metadata(output_path, trades, new_meta)

    return str(output_path)


def save_temp_trades(temp_dir: Path, trades: pd.DataFrame, asset: str,
                     filtered: bool, selected_day_types: list,
                     selected_trade_types) -> Path:
    """
    Write trades to {temp_dir}/{asset}_temp_file_{N}.parquet (first free N,
    starting at 1) with the filter kv-metadata attached.

    Filter-aware dedup, reuse-flavoured: when an existing temp file has the
    same row content AND the same filter metadata, no new file is written —
    the existing file's path is returned so the caller opens that one.
    """
    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)

    new_meta = build_filter_metadata(filtered, selected_day_types, selected_trade_types)

    for f in sorted(temp_path.glob(f"{asset}_temp_file_*.parquet")):
        if pd.read_parquet(f).equals(trades) and read_filter_metadata(f) == new_meta:
            return f

    n = 1
    while (output_path := temp_path / f"{asset}_temp_file_{n}.parquet").exists():
        n += 1

    write_with_metadata(output_path, trades, new_meta)
    return output_path
