from __future__ import annotations

import streamlit as st

from agent_monitor.analysis.ab_test import compare_pair
from components import render_run_sidebar, get_filtered_df

render_run_sidebar()

st.title("A/B Testing")
df = get_filtered_df()
agents = sorted(df.agent_name.unique())

if len(agents) < 2:
    st.warning("Need >=2 agents in this run for A/B testing")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    a = st.selectbox("Agent A", agents, index=0)
with c2:
    b = st.selectbox("Agent B", agents, index=1 if len(agents) > 1 else 0)

if a == b:
    st.warning("Pick two different agents")
    st.stop()

result = compare_pair(df, a, b)

if "error" in result:
    st.error(result["error"])
else:
    st.metric("p-value (latency)", f"{result.get('p_value_latency', 1.0):.4f}")
    st.metric("t-stat", f"{result.get('t_stat', 0.0):.4f}")

with st.expander("Raw test output"):
    st.json(result)
