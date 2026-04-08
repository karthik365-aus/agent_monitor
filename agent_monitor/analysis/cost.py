from __future__ import annotations

import pandas as pd
import yaml

from ..config import SETTINGS


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"total_cost": 0.0, "computed_from_tokens": False}

    try:
        raw = yaml.safe_load(SETTINGS.prices_config.read_text(encoding="utf-8")) or {}
        prices = raw.get("prices", {})
    except Exception:
        prices = {}

    work = df.copy()
    has_tokens = {"input_tokens", "output_tokens", "model"}.issubset(work.columns)
    if not has_tokens:
        if "cost" in work.columns:
            return {"total_cost": float(work.cost.fillna(0).sum()), "computed_from_tokens": False}
        return {"total_cost": 0.0, "computed_from_tokens": False}

    def row_cost(row: pd.Series) -> float:
        model_prices = prices.get(row.get("model"), {"in": 0.0, "out": 0.0})
        in_tokens = float(row.get("input_tokens", 0) or 0)
        out_tokens = float(row.get("output_tokens", 0) or 0)
        return (in_tokens * model_prices["in"] + out_tokens * model_prices["out"]) / 1_000_000

    work["computed_cost"] = work.apply(row_cost, axis=1)
    by_model = (
        work.groupby("model", as_index=False)
        .agg(total_cost=("computed_cost", "sum"), calls=("model", "count"))
        .sort_values("total_cost", ascending=False)
    )
    return {
        "total_cost": float(work["computed_cost"].sum()),
        "computed_from_tokens": True,
        "by_model": by_model,
    }
