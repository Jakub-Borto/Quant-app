# app: streamlit

import pandas as pd
import streamlit as st

df = pd.read_parquet("D:/market_data/regimes/ES/ES_vol_realized_z/2026-06-23.parquet")
print(df.columns)
print("asdasd")
st.dataframe(df)