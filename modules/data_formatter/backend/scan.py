"""
Input/output folder scanning for the Data Formatter.

Extracted verbatim from legacy_streamlit/views/data_formatter.py.

Change vs the old view: get_output_folders takes the data root explicitly
(the old code hardcoded data/parquet); the three-level hierarchy
{type}/{ASSET}/{dataset} is unchanged.
"""

from pathlib import Path


def get_structure(base_path: Path) -> dict:
    """
    Scans base_path and returns nested structure:
    { type: { asset: [folder_name, ...] } }
    """
    structure = {}
    if not base_path.exists():
        return structure
    for type_dir in sorted(base_path.iterdir()):
        if not type_dir.is_dir():
            continue
        structure[type_dir.name] = {}
        for asset_dir in sorted(type_dir.iterdir()):
            if not asset_dir.is_dir():
                continue
            datasets = sorted([f.name for f in asset_dir.iterdir() if f.is_dir()])
            if datasets:
                structure[type_dir.name][asset_dir.name] = datasets
    return structure


def get_output_folders(data_root: Path, asset_type: str, asset: str) -> list:
    out_path = Path(data_root) / "parquet" / asset_type / asset
    if not out_path.exists():
        return []
    return sorted([f.name for f in out_path.iterdir() if f.is_dir()])
