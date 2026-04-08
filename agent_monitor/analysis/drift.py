from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SETTINGS


def detect_all(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"status": "no_data", "rows": 0, "summary": pd.DataFrame(), "timeseries": pd.DataFrame()}

    required = {"agent_name", "hallucination_score", "latency"}
    if not required.issubset(df.columns):
        return {"status": "invalid_data", "rows": int(len(df)), "summary": pd.DataFrame(), "timeseries": pd.DataFrame()}

    work = df.copy()
    if "timestamp" in work.columns:
        work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
        work = work.sort_values("timestamp")
    else:
        work["timestamp"] = pd.RangeIndex(start=0, stop=len(work), step=1)

    threshold = float(SETTINGS.thresholds.drift_std_threshold)
    rows: list[dict] = []

    for agent in sorted(work["agent_name"].dropna().unique()):
        sub = work[work["agent_name"] == agent].copy()
        if len(sub) < max(6, SETTINGS.thresholds.sample_size_min * 2):
            rows.append(
                {
                    "agent_name": agent,
                    "samples": len(sub),
                    "hallucination_change_pct": np.nan,
                    "latency_change_pct": np.nan,
                    "hallucination_z": np.nan,
                    "latency_z": np.nan,
                    "drift_flag": False,
                    "reason": "insufficient samples",
                }
            )
            continue

        split = max(1, len(sub) // 2)
        base = sub.iloc[:split]
        recent = sub.iloc[split:]

        # --- Latency drift (use all rows; winsorize baseline std at p95 to avoid outlier inflation) ---
        l_base = float(base["latency"].mean())
        l_recent = float(recent["latency"].mean())
        l_base_cap = float(base["latency"].clip(upper=base["latency"].quantile(0.95)).std(ddof=0) or 0.0)
        l_z = (l_recent - l_base) / l_base_cap if l_base_cap > 1e-9 else 0.0
        l_change = ((l_recent - l_base) / l_base * 100.0) if abs(l_base) > 1e-9 else 0.0

        # Absolute latency threshold: flag if >100% increase regardless of z-score
        abs_latency_threshold = float(getattr(SETTINGS.thresholds, "drift_latency_abs_threshold", 100.0))
        latency_abs_flag = l_change > abs_latency_threshold and len(base) >= 3 and len(recent) >= 3

        # --- Hallucination drift (only on non-null rows to avoid null pollution) ---
        h_base_vals = base["hallucination_score"].dropna()
        h_recent_vals = recent["hallucination_score"].dropna()
        min_h_samples = max(3, SETTINGS.thresholds.sample_size_min // 2)
        if len(h_base_vals) >= min_h_samples and len(h_recent_vals) >= min_h_samples:
            h_base = float(h_base_vals.mean())
            h_recent = float(h_recent_vals.mean())
            h_std = float(h_base_vals.std(ddof=0) or 0.0)
            h_z = (h_recent - h_base) / h_std if h_std > 1e-9 else 0.0
            h_change = ((h_recent - h_base) / h_base * 100.0) if abs(h_base) > 1e-9 else 0.0
        else:
            h_base = float("nan")
            h_recent = float("nan")
            h_z = 0.0
            h_change = float("nan")

        drift_flag = abs(h_z) >= threshold or abs(l_z) >= threshold or latency_abs_flag
        reasons = []
        if abs(h_z) >= threshold:
            reasons.append("hallucination shift")
        if abs(l_z) >= threshold or latency_abs_flag:
            reasons.append(f"latency shift ({l_change:+.0f}%)")

        rows.append(
            {
                "agent_name": agent,
                "samples": len(sub),
                "hallucination_base": round(h_base, 3),
                "hallucination_recent": round(h_recent, 3),
                "hallucination_change_pct": round(h_change, 2),
                "hallucination_z": round(h_z, 2),
                "latency_base": round(l_base, 3),
                "latency_recent": round(l_recent, 3),
                "latency_change_pct": round(l_change, 2),
                "latency_z": round(l_z, 2),
                "drift_flag": bool(drift_flag),
                "reason": ", ".join(reasons) if reasons else "stable",
            }
        )

    summary = pd.DataFrame(rows).sort_values(["drift_flag", "hallucination_change_pct"], ascending=[False, False])

    timeseries = (
        work.groupby(["timestamp", "agent_name"], as_index=False)
        .agg(hallucination_score=("hallucination_score", "mean"), latency=("latency", "mean"))
        .sort_values("timestamp")
    )

    return {
        "status": "ok",
        "rows": int(len(work)),
        "threshold_z": threshold,
        "summary": summary,
        "timeseries": timeseries,
    }
