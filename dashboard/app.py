from __future__ import annotations

import streamlit as st

from components import render_run_sidebar

st.set_page_config(page_title="Agent Monitor", layout="wide", page_icon="🤖")

render_run_sidebar()

st.title("Agent Performance Monitor")
st.caption("Pick a page from the sidebar →")
