from __future__ import annotations

import plotly.express as px
import streamlit as st

from agent_monitor.analysis import drift
from components import render_run_sidebar, get_filtered_df

render_run_sidebar()

st.title("Drift")
df = get_filtered_df()
res = drift.detect_all(df)

if res.get("status") != "ok":
    st.warning(f"Drift unavailable: {res.get('status')}")
    st.stop()

summary = res["summary"]
st.caption(f"Rows analyzed: {res['rows']}  |  Drift threshold (z): {res['threshold_z']}")

if summary.empty:
    st.info("Not enough data to compute drift.")
    st.stop()

st.subheader("Per-agent drift summary")
st.dataframe(summary, width='stretch', hide_index=True)

drifted = summary[summary["drift_flag"] == True]  # noqa: E712
if drifted.empty:
    st.success("No significant drift detected.")
else:
    st.error(f"Detected drift in {len(drifted)} agent(s).")

ts = res["timeseries"]
if not ts.empty and "timestamp" in ts.columns:
    st.subheader("Hallucination trend")
    fig_h = px.line(
        ts,
        x="timestamp",
        y="hallucination_score",
        color="agent_name",
        markers=True,
    )
    st.plotly_chart(fig_h, width='stretch')

    st.subheader("Latency trend")
    fig_l = px.line(
        ts,
        x="timestamp",
        y="latency",
        color="agent_name",
        markers=True,
    )
    st.plotly_chart(fig_l, width='stretch')
