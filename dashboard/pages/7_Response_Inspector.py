from __future__ import annotations

import pandas as pd
import streamlit as st

from components import render_run_sidebar, get_filtered_df

render_run_sidebar()

st.title("Response Inspector")
df = get_filtered_df()

c1, c2, c3 = st.columns(3)
with c1:
    agent = st.selectbox("Agent", ["(all)"] + sorted(df.agent_name.unique()))
with c2:
    min_h = st.slider("Min hallucination %", 0, 100, 0)
with c3:
    only_errors = st.checkbox("Only errors")

view = df.copy()
if agent != "(all)":
    view = view[view.agent_name == agent]
view = view[
    (view.hallucination_score.isna()) | (view.hallucination_score >= min_h)
]
if only_errors and "status" in view:
    view = view[view.status == "error"]

view = view.sort_values("hallucination_score", ascending=False, na_position="last")
st.caption(f"{len(view)} responses")

for _, row in view.head(50).iterrows():
    h = row.hallucination_score
    h_label = "N/A" if pd.isna(h) else f"{h:.0f}%"
    with st.expander(f"[{h_label}] {row.agent_name} — {str(row.query)[:70]}"):
        st.markdown(f"**Query:** {row.query}")
        st.markdown(f"**Response:** {row.response}")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Hallucination", "N/A" if pd.isna(h) else f"{h:.1f}%")
        cc2.metric("Latency", f"{row.latency:.2f}s")
        cc3.metric("Model", row.model)
