# app: streamlit

import databento as db
import pandas as pd
import streamlit as st
import pyarrow.parquet as pq

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option('display.max_rows', 30)

@st.cache_data
def load_dbn(path, tz="America/New_York"):
    data = db.DBNStore.from_file(path)
    return data.to_df(tz=tz)


df_definition = load_dbn("D:/market_data/raw_dbn/Options_on_futures/ES/ES_2010_06_06-2026-07-02_DEFINITION/glbx-mdp3-20260625.definition.dbn.zst")
df_statistics = load_dbn("D:/market_data/raw_dbn/Options_on_futures/ES/ES_2010_06_06-2026-07-02_STATISTICS/glbx-mdp3-20260625.statistics.dbn.zst")
df_statistics_next_day = load_dbn("D:/market_data/raw_dbn/Options_on_futures/ES/ES_2010_06_06-2026-07-02_STATISTICS/glbx-mdp3-20260626.statistics.dbn.zst")

pf = pq.ParquetFile("D:/market_data/parquet/Futures/ES/ES_1m_advanced/2026-06-25.parquet")
metadata = pf.schema_arrow.metadata

front_month = metadata[b"front_month"].decode()
print(f"Front month: {front_month}")

# --- Definitions: filter by underlying ---
print(f"Definitions rows before filtering: {len(df_definition)}")
df_definition_filtered = df_definition[df_definition["underlying"] == front_month]
df_definition_filtered[["activation", "expiration"]] = df_definition_filtered[["activation", "expiration"]].apply(
    lambda col: col.dt.tz_convert("America/New_York")
)
print(f"Definitions rows after filtering: {len(df_definition_filtered)}")

# --- Statistics: filter by symbols left in definitions ---
symbols = df_definition_filtered["symbol"].unique()

print(f"Statistics rows before filtering: {len(df_statistics)}")
df_statistics_filtered = df_statistics[df_statistics["symbol"].isin(symbols)]
df_statistics_filtered = df_statistics_filtered[df_statistics_filtered["stat_type"] == 9]
print(f"Statistics rows after filtering: {len(df_statistics_filtered)}")

df_statistics_next_day_filtered = df_statistics_next_day[df_statistics_next_day["symbol"].isin(symbols)]
df_statistics_next_day_filtered = df_statistics_next_day_filtered[df_statistics_next_day_filtered["stat_type"] == 9]


st.write("Definition")
st.dataframe(df_definition_filtered)
st.write("Statistics")
st.dataframe(df_statistics_filtered)
st.write("Statistics next day")
st.dataframe(df_statistics_next_day_filtered)