from __future__ import annotations

import pandas as pd


def check(df: pd.DataFrame, hallucination_threshold: float = 40.0) -> list[str]:
    if df.empty:
        return ["no data"]
    alerts = []
    high = df[df.hallucination_score > hallucination_threshold]
    if not high.empty:
        alerts.append(f"high hallucination count={len(high)}")
    return alerts
