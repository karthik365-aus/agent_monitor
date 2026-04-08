from __future__ import annotations

import pandas as pd


def overview(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"status": "no_data"}
    return {"rows": int(len(df)), "agents": int(df.agent_name.nunique())}
