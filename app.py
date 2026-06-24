import streamlit as st

st.set_page_config(
    page_title="Quant Research Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "page" not in st.session_state:
    st.session_state.page = "home"

from views import home, data_formatter, backtester, analytics, monte_carlo

def router():
    page = st.session_state.page

    if page == "home":
        home.render()
    elif page == "data_formatter":
        data_formatter.render()
    elif page == "backtester":
        backtester.render()
    elif page == "analytics":
        analytics.render()
    elif page == "monte_carlo":
        monte_carlo.render()
    else:
        st.session_state.page = "home"
        st.rerun()

router()



