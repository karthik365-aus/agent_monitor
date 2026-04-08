from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components import render_run_sidebar, get_filtered_df, agent_filter, status_badge

render_run_sidebar()

st.title("Overview")
df = get_filtered_df()
selected = agent_filter(df)
df = df[df.agent_name.isin(selected)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total calls", len(df))
_hm = df.hallucination_score.mean()
c2.metric(
    "Avg accuracy",
    f"{100 - _hm:.1f}%" if pd.notna(_hm) else "N/A",
)
c3.metric("Avg latency", f"{df.latency.mean():.2f}s")
err_rate = (df.status == "error").mean() * 100 if "status" in df else 0
c4.metric("Error rate", f"{err_rate:.1f}%")

st.divider()
st.subheader("Health by agent")
health_rows = []
for agent in df.agent_name.unique():
    a = df[df.agent_name == agent]
    am = a.hallucination_score.mean()
    accuracy = 100 - am if pd.notna(am) else float("nan")
    speed = max(0.0, 100 - a.latency.mean() * 20)
    std = a.hallucination_score.std() if len(a) > 1 else 0
    consistency = max(0.0, 100 - (0 if pd.isna(std) else std))
    _ov = accuracy * 0.4 + speed * 0.3 + consistency * 0.3
    overall = _ov if pd.notna(_ov) else float("nan")
    health_rows.append(
        {
            "Agent": agent,
            "Calls": len(a),
            "Accuracy": round(accuracy, 1) if pd.notna(accuracy) else None,
            "Latency": round(a.latency.mean(), 2),
            "Overall": round(overall, 1) if pd.notna(overall) else None,
            "Status": status_badge(overall) if pd.notna(overall) else "—",
        }
    )
st.dataframe(health_rows, width='stretch', hide_index=True)

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("Hallucination by agent")
    fig = px.bar(
        df.groupby("agent_name", as_index=False).hallucination_score.mean(),
        x="agent_name",
        y="hallucination_score",
        color="hallucination_score",
        color_continuous_scale="RdYlGn_r",
    )
    st.plotly_chart(fig, width='stretch')

with right:
    st.subheader("Latency distribution")
    fig = px.box(df, x="agent_name", y="latency", color="agent_name")
    st.plotly_chart(fig, width='stretch')

if "timestamp" in df:
    st.subheader("Accuracy over time")
    trend = df.assign(accuracy=100 - df.hallucination_score).sort_values("timestamp")
    fig = px.line(trend, x="timestamp", y="accuracy", color="agent_name", markers=True)
    st.plotly_chart(fig, width='stretch')
