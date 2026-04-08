from __future__ import annotations

import pandas as pd
from scipy.stats import ttest_ind


def compare_pair(df: pd.DataFrame, a: str, b: str) -> dict:
    da = df[df.agent_name == a]
    db = df[df.agent_name == b]
    if len(da) < 5 or len(db) < 5:
        return {"error": "insufficient data (need ≥5 samples each)", "a": a, "b": b}
    t_stat, p_value = ttest_ind(da.latency, db.latency, equal_var=False)
    return {"a": a, "b": b, "p_value_latency": float(p_value), "t_stat": float(t_stat)}


def compare_all(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    names = sorted(df.agent_name.unique())
    return [compare_pair(df, a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
