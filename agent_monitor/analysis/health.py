from __future__ import annotations

import pandas as pd


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"status": "no_data"}
    return {
        "status": "ok",
        "rows": int(len(df)),
        "agents": int(df.agent_name.nunique()),
        "error_rate": float((df.status == "error").mean()) if "status" in df else 0.0,
    }
