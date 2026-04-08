from __future__ import annotations

import yaml
import pandas as pd
import streamlit as st

from agent_monitor.config import SETTINGS
from agent_monitor.storage.store import get_metrics_df


@st.cache_data(ttl=300)
def _load_excluded_agents() -> frozenset[str]:
    """Agents that should not appear in analysis (judges, disabled-for-collection)."""
    try:
        cfg = yaml.safe_load(SETTINGS.agents_config.read_text(encoding="utf-8")) or {}
        return frozenset(
            a["name"]
            for a in cfg.get("agents", [])
            if a.get("role") == "judge" or not a.get("enabled_for_collection", True)
        )
    except Exception:
        return frozenset()


@st.cache_data(ttl=30)
def load_runs() -> pd.DataFrame:
    df = get_metrics_df()
    if df.empty or "run_id" not in df.columns:
        return pd.DataFrame()
    runs = (
        df[df.run_id.notna()]
        .groupby("run_id", as_index=False)
        .agg(started=("timestamp", "min"), n=("id", "count"), agents=("agent_name", "nunique"))
        .sort_values("started", ascending=False)
    )
    return runs


def render_run_sidebar() -> None:
    st.sidebar.title("🤖 Agent Monitor")
    runs = load_runs()
    if runs.empty:
        st.sidebar.warning("No runs yet")
        st.sidebar.caption("Go to **Run Collection** to start one")
        st.session_state["run_id"] = None
    else:
        options = ["(all runs)"] + runs["run_id"].tolist()
        pick = st.sidebar.selectbox(
            "Run",
            options,
            key="run_picker",
            format_func=lambda r: "all runs"
            if r == "(all runs)"
            else f"{r}  ·  {runs.loc[runs.run_id == r, 'n'].iloc[0]} calls",
        )
        st.session_state["run_id"] = None if pick == "(all runs)" else pick

    st.sidebar.divider()
    if st.sidebar.button("🔄 Refresh data", key="refresh_data"):
        st.cache_data.clear()
        st.rerun()


def get_filtered_df(min_calls: int = 3) -> pd.DataFrame:
    run_id = st.session_state.get("run_id")
    df = get_metrics_df(run_id=run_id)
    if df.empty:
        st.info("No data for this selection. Run a collection first.")
        st.stop()

    # Drop judge agents and any agents disabled for collection
    excluded = _load_excluded_agents()
    if excluded:
        df = df[~df["agent_name"].isin(excluded)]

    # Drop phantom/test agents that have too few calls in this selection
    if min_calls > 1 and "agent_name" in df.columns:
        counts = df["agent_name"].value_counts()
        valid = counts[counts >= min_calls].index
        df = df[df["agent_name"].isin(valid)]

    if df.empty:
        st.info("No data for this selection. Run a collection first.")
        st.stop()
    return df


def agent_filter(df: pd.DataFrame) -> list[str]:
    agents = sorted(df["agent_name"].dropna().unique().tolist())
    return st.multiselect("Agents", agents, default=agents)


def metric_card(label: str, value: str, delta: str | None = None, help: str | None = None) -> None:
    st.metric(label, value, delta=delta, help=help)


def status_badge(score: float) -> str:
    if score >= 85:
        return "🟢 Healthy"
    if score >= 70:
        return "🟡 Warning"
    return "🔴 Critical"
