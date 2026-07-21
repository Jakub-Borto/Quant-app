# app: streamlit

import databento as db
import pandas as pd
import streamlit as st
import pyarrow.parquet as pq

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option('display.max_rows', 30)



'''
data = db.DBNStore.from_file("D:/Quant_app/data/raw_dbn/Futures/30_assets/30_assets_ohlcv/glbx-mdp3-20100606-20260516.ohlcv-1m.dbn.zst")
df = next(data.to_df(count=10000))
print(df)
'''



'''
data2 = db.DBNStore.from_file("D:/Quant_app/data/tests/glbx-mdp3-20260514.mbo.dbn.zst")
df2 = data2.to_df()
print(df2.head(20))
st.dataframe(df2)
'''




'''

'''
df3 = pd.read_parquet("D:/market_data/parquet/Futures/ES/ES_1m_indicators/2025-04-23.parquet")
print(df3.head(20))
print(df3.dtypes)
print(df3.columns)
st.dataframe(df3)