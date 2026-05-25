import streamlit as st


def go(page: str):
    st.session_state.page = page
    st.rerun()


def render():
    st.title("Research Engine")
    st.caption("ES · NQ · Intraday Futures")

    st.write("")
    st.write("")

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.subheader("01 · Data Formatter")
        st.write("Convert raw DBN files into enriched 1m candles stored as Parquet.")
        if st.button("Open →", key="nav_data", use_container_width=True):
            go("data_formatter")
        st.write("")
        st.subheader("04 · Footprint Chart")
        st.write("See an actual chart with footprint data and passive orders")
        if st.button("Open →", key="nav_chart", use_container_width=True):
            go("chart")
        

    with col2:
        st.subheader("02 · Backtester")
        st.write("Run vectorized strategies on your datasets. Outputs trades to Parquet.")
        if st.button("Open →", key="nav_backtest", use_container_width=True):
            go("backtester")
        st.write("")
        st.subheader("05 · Monte Carlo")
        st.write("Run Monte Carlo simulations to stress test the strategy")
        if st.button("Open →", key="nav_monte_carlo", use_container_width=True):
            go("monte_carlo")

    with col3:
        st.subheader("03 · Analytics")
        st.write("Load trades, apply position sizing, explore equity curve and metrics.")
        if st.button("Open →", key="nav_analytics", use_container_width=True):
            go("analytics")
        st.write("")
        st.subheader("06 · Simple Chart")
        st.write("See a simple chart with multiple timeframes")
        if st.button("Open →", key="nav_simple_chart", use_container_width=True):
            go("simple_chart")
