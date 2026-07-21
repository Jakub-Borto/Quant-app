# app: streamlit
"""Example Streamlit quick script.

The `# app: streamlit` comment on line 1 (anywhere in the first 30 lines
works, `STREAMLIT = True` is also accepted) tells the Scripts module to
launch this file with `streamlit run` on a free port and open it in the
browser. Without a marker a script runs as plain `python -u` with its
output in the module's console.

Repo imports: the process's cwd is this script's folder, NOT the repo root
(the root's scratch inspect.py would shadow stdlib `inspect` and break
numpy). Use the sys.path.APPEND idiom below — append lands after the stdlib
paths, so nothing gets shadowed. Scripts living in an external extra folder
(added via Settings) must hardcode the repo path instead of parents[1].
"""

import sys
from pathlib import Path

_repo = str(Path(__file__).resolve().parents[1])
if _repo not in sys.path:
    sys.path.append(_repo)

import numpy as np
import pandas as pd
import streamlit as st

from modules.common.backend.settings import load_settings

st.set_page_config(page_title="Example — Hello", layout="wide")
st.title("Hello from the Scripts module")

settings = load_settings()
st.caption("Configured data roots:")
st.write([str(r) for r in settings.data_roots])

st.subheader("Random walk demo")
n = st.slider("Bars", 50, 2000, 500)
prices = 5000 + np.cumsum(np.random.randn(n))
st.line_chart(pd.Series(prices, name="price"))
