from __future__ import annotations

import pandas as pd


def render(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"status": "no_data"}
    return {
        "status": "ok",
        "latency_mean": float(df.latency.mean()),
        "halluc_mean": float(df.hallucination_score.mean()),
    }
