# app: streamlit

import databento as db
import pandas as pd
import streamlit as st
import os
import pyarrow.parquet as pq

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option('display.max_rows', 30)

data4 = pd.read_parquet("D:/market_data/parquet/Options_on_futures/ES/ES_options_oi_snapshots/2026-03-24.parquet")
st.dataframe(data4)





@st.cache_data
def load_parquet(path, mtime=None):
    # mtime is only part of the cache key: the cache invalidates when the
    # file is rewritten by a transform re-run
    return pd.read_parquet(path)


_p5 = "D:/market_data/parquet/Options_on_futures/ES/ES_options_5m/2026-03-24.parquet"
data5 = load_parquet(_p5, os.path.getmtime(_p5))
st.dataframe(data5.head(10000))


# --- ES_options_5m contracts dict (parquet footer metadata) ---
import json

meta5 = pq.ParquetFile(
    "D:/market_data/parquet/Options_on_futures/ES/ES_options_5m/2026-03-24.parquet"
).schema_arrow.metadata
contracts = json.loads(meta5[b"contracts"])

st.write(f"ES_options_5m 2026-03-24 — contracts dict "
         f"({len(contracts)} contracts, multiplier {meta5[b'multiplier'].decode()})")
chosen_id = st.text_input("instrument_id (leave empty to see the entire dict)", "").strip()

if not chosen_id:
    st.dataframe(pd.DataFrame.from_dict(contracts, orient="index"))
else:
    info = contracts.get(chosen_id)
    if info is None:
        st.write(f"instrument_id {chosen_id} is not in this file's dict")
    else:
        st.json(info)
        st.write(f"5m rows for {chosen_id} — {info['underlying']} {info['series']} "
                 f"K={info['strike']} {info['cp_flag']} exp {info['expiry']}")
        st.dataframe(data5[data5["instrument_id"] == int(chosen_id)])


# --- full history of one instrument across a date range of ES_options_5m ---
import glob
from pathlib import Path

RANGE_ID = 42752324
RANGE_START = pd.Timestamp("2026-02-13 16:00:00-05:00")
RANGE_END = pd.Timestamp("2026-03-25 16:00:00-04:00")


@st.cache_data
def load_instrument_range(iid, t0, t1, newest_mtime=None):
    folder = "D:/market_data/parquet/Options_on_futures/ES/ES_options_5m"
    d0, d1 = str(t0.date()), str((t1 + pd.Timedelta(days=1)).date())
    parts = []
    info = None
    for f in sorted(glob.glob(folder + "/*.parquet")):
        stem = Path(f).stem
        if not stem[0].isdigit() or not (d0 <= stem <= d1):
            continue
        d = pd.read_parquet(f)
        d = d[d["instrument_id"] == iid]
        if len(d):
            parts.append(d)
            if info is None:
                meta = pq.ParquetFile(f).schema_arrow.metadata
                info = json.loads(meta[b"contracts"]).get(str(iid))
    if not parts:
        return pd.DataFrame(), info
    out = pd.concat(parts).sort_index(kind="stable")
    return out[(out.index >= t0) & (out.index <= t1)], info


_5m_dir = "D:/market_data/parquet/Options_on_futures/ES/ES_options_5m"
_newest = max(os.path.getmtime(f) for f in glob.glob(_5m_dir + "/*.parquet"))
range_df, range_info = load_instrument_range(RANGE_ID, RANGE_START, RANGE_END,
                                             _newest)
st.write(f"instrument_id {RANGE_ID} — {RANGE_START} to {RANGE_END}: "
         f"{len(range_df)} rows")
if range_info:
    st.json(range_info)
st.dataframe(range_df, height=600)