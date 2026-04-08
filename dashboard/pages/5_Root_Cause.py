from __future__ import annotations

import plotly.express as px
import streamlit as st

from agent_monitor.analysis import root_cause
from components import render_run_sidebar, get_filtered_df

render_run_sidebar()

st.title("Root Cause")
df = get_filtered_df()
res = root_cause.analyze_all(df)

if res.get("status") != "ok":
    st.warning(f"Root-cause unavailable: {res.get('status')}")
    st.stop()

st.metric("Error count", res["error_count"])

agent_summary = res["agent_summary"]
high_risk = res["high_risk_responses"]

st.subheader("Agent issue summary")
st.dataframe(agent_summary, width='stretch', hide_index=True)

if not agent_summary.empty:
    fig = px.bar(
        agent_summary,
        x="agent_name",
        y="severity_score",
        color="primary_issue",
        title="Severity by agent",
    )
    st.plotly_chart(fig, width='stretch')

st.subheader("Top high-risk responses")
st.dataframe(
    high_risk[["agent_name", "query", "hallucination_score", "latency", "status", "risk_score"]],
    width='stretch',
    hide_index=True,
)

for _, row in high_risk.head(20).iterrows():
    with st.expander(f"{row['agent_name']} | risk={row['risk_score']:.1f} | {str(row['query'])[:80]}"):
        st.markdown(f"**Query:** {row['query']}")
        st.markdown(f"**Response:** {row['response']}")
