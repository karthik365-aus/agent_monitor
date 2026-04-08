from __future__ import annotations

import pandas as pd

from ..config import SETTINGS
from .normalization import latency_to_speed_score


def render(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Agent", "Overall Score"])
    w = SETTINGS.weights.leaderboard
    latency_scale = SETTINGS.weights.latency_score_scale
    rows = []
    for agent in sorted(df.agent_name.unique()):
        sub = df[df.agent_name == agent]
        hm = sub.hallucination_score.mean()
        acc = 100 - hm if pd.notna(hm) else float("nan")
        speed = latency_to_speed_score(sub.latency.mean(), scale=latency_scale)
        std = sub.hallucination_score.std() if len(sub) > 1 else 0
        consistency = 100 - (0 if pd.isna(std) else std)
        weighted = acc * w.accuracy + speed * w.speed + consistency * w.consistency
        if pd.isna(weighted):
            overall = float("nan")
        else:
            overall = max(0.0, min(100.0, weighted))
        rows.append(
            {
                "Agent": agent,
                "Overall Score": round(overall, 2) if pd.notna(overall) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("Overall Score", ascending=False)
