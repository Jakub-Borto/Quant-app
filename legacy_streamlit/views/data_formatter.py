import streamlit as st
from pathlib import Path
import importlib.util
import time


def go(page: str):
    st.session_state.page = page
    st.rerun()


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


def get_output_folders(asset_type: str, asset: str) -> list:
    out_path = Path("data/parquet") / asset_type / asset
    if not out_path.exists():
        return []
    return sorted([f.name for f in out_path.iterdir() if f.is_dir()])


def get_transforms():
    transforms_path = Path("transforms")
    if not transforms_path.exists():
        return []
    return sorted([
        f.stem for f in transforms_path.glob("*.py")
        if f.stem not in ["__init__", "base"]
    ])


def load_transform(name: str):
    path = Path("transforms") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render():
    if st.button("← Back"):
        go("home")

    st.title("Data Formatter")
    st.caption("Convert raw DBN files into candles and save as Parquet.")
    st.write("")

    transforms = get_transforms()
    if not transforms:
        st.error("No transform scripts found in transforms/")
        return

    col1, col2 = st.columns(2)

    with col1:
        # input source selector
        input_source = st.selectbox("Input source", ["raw_dbn", "parquet"])
        base_path    = Path("data") / input_source

        structure = get_structure(base_path)
        if not structure:
            st.error(f"No folders found in data/{input_source}")
            return

        # type selector
        asset_types = list(structure.keys())
        asset_type  = st.selectbox("Type", asset_types, key=f"type_{input_source}")

        # asset selector
        assets = list(structure.get(asset_type, {}).keys())
        if not assets:
            st.error(f"No assets found under {asset_type}")
            return
        asset = st.selectbox("Asset", assets, key=f"asset_{input_source}")

        # dataset selector
        datasets = structure[asset_type].get(asset, [])
        if not datasets:
            st.error(f"No datasets found under {asset_type}/{asset}")
            return
        dataset = st.selectbox("Input dataset", datasets, key=f"dataset_{input_source}_{asset_type}_{asset}")

        # transform selector
        transform_name = st.selectbox("Transform", transforms)

    with col2:
        st.info(f"Output path: `data/parquet/{asset_type}/{asset}/`")

        NEW_FOLDER       = "── New folder ──"
        existing_folders = get_output_folders(asset_type, asset)
        output_options   = [NEW_FOLDER] + existing_folders

        output_selection = st.selectbox(
            "Output folder",
            output_options,
            key=f"output_{input_source}_{asset_type}_{asset}"
        )

        if output_selection == NEW_FOLDER:
            new_folder_name    = st.text_input("New folder name", placeholder="e.g. ES_indicators")
            output_folder_name = new_folder_name.strip()
        else:
            output_folder_name = output_selection

    st.write("")
    skip_existing = st.checkbox("Skip already processed files", value=True)
    st.write("")

    _, _, btn_col, _, _ = st.columns(5)
    with btn_col:
        run = st.button("Run", type="primary", use_container_width=True)

    if run:
        if not output_folder_name:
            st.error("Please enter an output folder name.")
            return

        input_path  = str(base_path / asset_type / asset / dataset)
        output_path = str(Path("data/parquet") / asset_type / asset / output_folder_name)

        transform = load_transform(transform_name)

        progress_bar  = st.progress(0)
        log_container = st.container(height=500)
        log_box       = log_container.empty()
        logs          = []

        def on_progress(current, total, message: str = ""):
            progress_bar.progress(current / total)
            if message:
                logs.append(message)
            log_box.code("\n".join(logs), language=None)

        start = time.time()

        transform.run_all(
            input_folder=input_path,
            output_folder=output_path,
            skip_existing=skip_existing,
            on_progress=on_progress,
        )

        elapsed = time.time() - start
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        if minutes > 0:
            st.success(f"Done in {minutes}m {seconds}s. Saved to data/parquet/{asset_type}/{asset}/{output_folder_name}")
        else:
            st.success(f"Done in {seconds}s. Saved to data/parquet/{asset_type}/{asset}/{output_folder_name}")