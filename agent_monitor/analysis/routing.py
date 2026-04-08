from __future__ import annotations

import pandas as pd

from ..config import SETTINGS
from .normalization import latency_to_penalty


def recommend(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"status": "no_data"}
    grouped = df.groupby("agent_name", as_index=False).agg(
        latency=("latency", "mean"),
        halluc=("hallucination_score", "mean"),
    )
    w = SETTINGS.weights.routing
    latency_scale = SETTINGS.weights.latency_score_scale
    grouped["latency_penalty"] = grouped["latency"].apply(
        lambda x: latency_to_penalty(x, scale=latency_scale)
    )
    grouped["score"] = grouped["halluc"] * w.hallucination + grouped["latency_penalty"] * w.latency
    best = grouped.sort_values("score").iloc[0]
    return {
        "best_agent": best.agent_name,
        "score": float(best.score),
        "latency_penalty": float(best.latency_penalty),
    }
