"""GSF - Geology Source Finder.  Entry point for Streamlit."""
from __future__ import annotations

import streamlit as st

# Page config must be the first Streamlit call
st.set_page_config(
    page_title="GSF - Geology Source Finder",
    page_icon="\U0001FAA8",
    layout="wide",
    initial_sidebar_state="expanded",
)

from gsf import main  # noqa: E402

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("An unexpected error occurred.")
        st.exception(e)
