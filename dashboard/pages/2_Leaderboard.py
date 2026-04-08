from __future__ import annotations

import streamlit as st

from agent_monitor.analysis import leaderboard
from components import render_run_sidebar, get_filtered_df, agent_filter

render_run_sidebar()

st.title("Leaderboard")
df = get_filtered_df()
selected = agent_filter(df)
sub = df[df.agent_name.isin(selected)]

lb = leaderboard.render(sub)
st.dataframe(lb, width='stretch', hide_index=True)
