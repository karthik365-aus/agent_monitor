from __future__ import annotations

import pandas as pd

from ..config import SETTINGS
from .normalization import latency_to_penalty


def analyze_all(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "status": "no_data",
            "error_count": 0,
            "agent_summary": pd.DataFrame(),
            "high_risk_responses": pd.DataFrame(),
        }

    work = df.copy()
    has_status = "status" in work.columns
    if not has_status:
        work["status"] = "success"

    error_df = work[work["status"] != "success"]
    high_h_threshold = float(SETTINGS.thresholds.failure_hallucination)
    latency_warn = float(SETTINGS.thresholds.latency_warning)

    agg = (
        work.groupby("agent_name", as_index=False)
        .agg(
            calls=("id", "count") if "id" in work.columns else ("agent_name", "count"),
            avg_hallucination=("hallucination_score", "mean"),
            p95_hallucination=("hallucination_score", lambda x: x.quantile(0.95)),
            avg_latency=("latency", "mean"),
            p95_latency=("latency", lambda x: x.quantile(0.95)),
            error_count=("status", lambda x: (x != "success").sum()),
        )
    )
    agg["error_rate_pct"] = (agg["error_count"] / agg["calls"] * 100).round(2)

    def reason(row: pd.Series) -> str:
        reasons: list[str] = []
        if row["avg_hallucination"] >= high_h_threshold:
            reasons.append("high hallucination")
        if row["avg_latency"] >= latency_warn:
            reasons.append("high latency")
        if row["error_rate_pct"] > 0:
            reasons.append("runtime errors")
        return ", ".join(reasons) if reasons else "stable"

    agg["primary_issue"] = agg.apply(reason, axis=1)
    latency_scale = SETTINGS.weights.latency_score_scale
    severity_w = SETTINGS.weights.root_cause.severity
    risk_w = SETTINGS.weights.root_cause.risk
    agg["latency_penalty"] = agg["avg_latency"].apply(
        lambda x: latency_to_penalty(x, scale=latency_scale)
    )
    agg["severity_score"] = (
        (agg["avg_hallucination"].clip(lower=0) * severity_w.hallucination)
        + (agg["latency_penalty"].clip(lower=0) * severity_w.latency)
        + (agg["error_rate_pct"].clip(lower=0) * severity_w.errors)
    ).round(2)
    agg = agg.sort_values("severity_score", ascending=False)

    risk = work.copy()
    risk["latency_penalty"] = risk["latency"].fillna(0).apply(
        lambda x: latency_to_penalty(x, scale=latency_scale)
    )
    risk["risk_score"] = (
        risk["hallucination_score"].fillna(0) * risk_w.hallucination
        + risk["latency_penalty"] * risk_w.latency
        + (risk["status"] != "success").astype(int) * risk_w.error_penalty
    )
    _risk_cols = [
        "agent_name", "query", "response", "hallucination_score",
        "latency", "status", "risk_score", "timestamp",
    ]
    high_risk = risk.sort_values("risk_score", ascending=False).head(50)[
        [c for c in _risk_cols if c in risk.columns]
    ]

    return {
        "status": "ok",
        "error_count": int(len(error_df)),
        "agent_summary": agg,
        "high_risk_responses": high_risk,
    }
