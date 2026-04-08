from __future__ import annotations

import streamlit as st

from components import render_run_sidebar

from agent_monitor.config import SETTINGS
from agent_monitor.agents.registry import AgentRegistry
from agent_monitor.collection.query_pool import QueryPool
from agent_monitor.collection.runner import build_plan, collect

render_run_sidebar()

st.title("Run Collection")


@st.cache_resource
def load_resources():
    return (
        AgentRegistry.from_yaml(str(SETTINGS.agents_config)),
        QueryPool.from_yaml(str(SETTINGS.queries_config)),
    )


registry, pool = load_resources()

with st.form("collect"):
    c1, c2 = st.columns(2)
    with c1:
        names = registry.names()
        agents = st.multiselect("Agents", names, default=names)
        n = st.slider("Queries per agent", 1, 200, 20)
        category = st.selectbox("Category", ["(auto by role)"] + list(pool.categories.keys()))
    with c2:
        parallel = st.slider("Parallel workers", 1, 16, 4)
        seed = st.number_input("Seed (optional)", value=0, step=1)
        shared = st.checkbox("Shared queries (for fair A/B)", value=False)
        label = st.text_input("Label", placeholder="e.g. nightly, prompt-v2")

    preview = st.form_submit_button("Preview plan")
    submit = st.form_submit_button("🚀 Run", type="primary")

if preview or submit:
    plan = build_plan(
        registry,
        pool,
        n_per_agent=n,
        agents=agents or None,
        category=None if category == "(auto by role)" else category,
        seed=seed or None,
        shared_queries=shared,
        label=label or None,
    )
    st.info(f"Plan: **{len(plan)} calls** across **{len({a.name for a, _ in plan.items})} agents**")

    if submit:
        progress = st.progress(0.0, text="Starting...")
        done = {"n": 0}

        def on_result(r):
            done["n"] += 1
            progress.progress(done["n"] / len(plan), text=f"{done['n']}/{len(plan)}  ·  {r.agent}")

        run_id = collect(plan, parallel=parallel, on_result=on_result, progress=False)
        st.success(f"✅ Complete — run_id `{run_id}`")
        st.cache_data.clear()
        st.balloons()
