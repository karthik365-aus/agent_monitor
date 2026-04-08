"""Run after every collection to catch data-quality regressions.

Usage:
    python validate.py                    # check all data
    python validate.py --run run_abc123   # check one run
    python validate.py --strict           # exit 1 on any issue (for CI)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import pandas as pd

from agent_monitor.config import SETTINGS
from agent_monitor.storage.store import get_metrics_df, get_metric_scores_df
from agent_monitor.agents.registry import AgentRegistry
from agent_monitor.analysis import leaderboard


@dataclass
class Issue:
    severity: str   # "error" | "warning" | "info"
    category: str
    message: str

    def __str__(self):
        icons = {"error": "🔴", "warning": "🟡", "info": "🔵"}
        return f"{icons[self.severity]} [{self.category}] {self.message}"


def validate(run_id: str | None = None) -> list[Issue]:
    issues: list[Issue] = []
    df = get_metrics_df(run_id=run_id)

    if df.empty:
        return [Issue("error", "data", "no metrics in database")]

    # ──────────────────────────────────────────────────────────────────
    # Phantom agents — anything in the DB that's not in agents.yaml
    # Catches the "a" agent bug from the screenshots
    # ──────────────────────────────────────────────────────────────────
    try:
        registry = AgentRegistry.from_yaml(str(SETTINGS.agents_config))
        known_agents = set(registry.names())
    except Exception as e:
        issues.append(Issue("error", "config", f"could not load agents.yaml: {e}"))
        known_agents = set()

    db_agents = set(df["agent_name"].dropna().unique())
    phantom = db_agents - known_agents
    if phantom:
        issues.append(Issue(
            "error", "phantom_agent",
            f"agents in DB but not in agents.yaml: {sorted(phantom)} — "
            f"likely test data, run: DELETE FROM metrics WHERE agent_name IN {tuple(phantom)}"
        ))

    # ──────────────────────────────────────────────────────────────────
    # Judge contamination — judge should not appear in collection runs
    # ──────────────────────────────────────────────────────────────────
    judges = {name for name in known_agents
              if "judge" in name.lower()}  # crude but effective
    judge_calls = df[df["agent_name"].isin(judges)]
    if not judge_calls.empty:
        issues.append(Issue(
            "warning", "judge_contamination",
            f"judge agents appear in metrics ({len(judge_calls)} rows). "
            f"build_plan should exclude role='judge'."
        ))

    # ──────────────────────────────────────────────────────────────────
    # Score health — null/zero/all-same patterns
    # ──────────────────────────────────────────────────────────────────
    null_scores = df["hallucination_score"].isna().sum()
    null_pct = null_scores / len(df) * 100
    if null_pct > 50:
        issues.append(Issue(
            "warning", "scoring",
            f"{null_scores}/{len(df)} ({null_pct:.0f}%) rows have null hallucination_score. "
            f"Likely the judge is failing — check metric_scores rationale column."
        ))

    non_null = df[df["hallucination_score"].notna()]
    if len(non_null) > 10:
        zero_scores = (non_null["hallucination_score"] == 0).sum()
        if zero_scores / len(non_null) > 0.9:
            issues.append(Issue(
                "warning", "scoring",
                f">90% of scored rows are exactly 0 — judge model is too lenient. "
                f"Try llama-3.3-70b-versatile instead of llama-3.1-8b-instant."
            ))

        if non_null["hallucination_score"].nunique() == 1:
            issues.append(Issue(
                "error", "scoring",
                f"all hallucination scores are identical "
                f"({non_null['hallucination_score'].iloc[0]}) — scorer is broken"
            ))

    # ──────────────────────────────────────────────────────────────────
    # Latency sanity — anything over 10s median is suspicious
    # ──────────────────────────────────────────────────────────────────
    by_agent = df[df["status"] == "success"].groupby("agent_name")["latency"].median()
    slow_agents = by_agent[by_agent > 10]
    if not slow_agents.empty:
        for agent, lat in slow_agents.items():
            issues.append(Issue(
                "error", "latency",
                f"{agent} median latency = {lat:.1f}s "
                f"— likely missing timeout in provider _call()"
            ))

    # Outliers within an agent (detected via p99 vs median)
    for agent, sub in df[df["status"] == "success"].groupby("agent_name"):
        if len(sub) >= 10:
            p50 = sub["latency"].median()
            p99 = sub["latency"].quantile(0.99)
            if p99 > p50 * 20 and p99 > 5:
                issues.append(Issue(
                    "warning", "latency",
                    f"{agent} has latency outliers: p50={p50:.2f}s p99={p99:.2f}s "
                    f"— retry storms or rate limiting"
                ))

    # ──────────────────────────────────────────────────────────────────
    # Error rate — should be under 5% in healthy operation
    # ──────────────────────────────────────────────────────────────────
    if "status" in df.columns:
        error_rate = (df["status"] == "error").mean() * 100
        if error_rate > 15:
            issues.append(Issue(
                "error", "errors",
                f"error rate = {error_rate:.1f}% — investigate with: "
                f"SELECT agent_name, error, COUNT(*) FROM metrics "
                f"WHERE status='error' GROUP BY agent_name, error"
            ))
        elif error_rate > 5:
            issues.append(Issue(
                "warning", "errors",
                f"error rate = {error_rate:.1f}% (target <5%)"
            ))

    # ──────────────────────────────────────────────────────────────────
    # Leaderboard sanity — judge should not be ranked
    # ──────────────────────────────────────────────────────────────────
    try:
        lb = leaderboard.render(df)
        if not lb.empty and any("judge" in a.lower() for a in lb["Agent"].astype(str)):
            issues.append(Issue(
                "error", "leaderboard",
                "judge agent appears in leaderboard ranking — analysis modules "
                "should filter out role='judge'"
            ))
    except Exception as e:
        issues.append(Issue("warning", "leaderboard", f"render failed: {e}"))

    # ──────────────────────────────────────────────────────────────────
    # Multi-scorer storage — metric_scores should have data if engine ran
    # ──────────────────────────────────────────────────────────────────
    try:
        scores_df = get_metric_scores_df(run_id=run_id)
        if scores_df.empty and not non_null.empty:
            issues.append(Issue(
                "warning", "storage",
                "metric_scores table is empty but hallucination_score has values — "
                "the engine is computing detailed scores but they're not being persisted"
            ))
        elif not scores_df.empty:
            scorers_used = set(scores_df["scorer_name"].unique())
            expected = set(SETTINGS.metrics.get("scorers", []))
            missing = expected - scorers_used
            if missing:
                issues.append(Issue(
                    "info", "storage",
                    f"configured scorers not in metric_scores: {missing}"
                ))
    except Exception as e:
        issues.append(Issue("info", "storage", f"could not query metric_scores: {e}"))

    # ──────────────────────────────────────────────────────────────────
    # Sample size per agent — analysis needs minimum
    # ──────────────────────────────────────────────────────────────────
    min_samples = SETTINGS.thresholds.sample_size_min
    counts = df.groupby("agent_name").size()
    too_small = counts[counts < min_samples]
    if not too_small.empty:
        for agent, n in too_small.items():
            issues.append(Issue(
                "warning", "samples",
                f"{agent} has only {n} samples (min={min_samples} for analysis)"
            ))

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="filter to a specific run_id")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 if any error or warning")
    parser.add_argument("--errors-only", action="store_true",
                        help="exit 1 only on errors, not warnings")
    args = parser.parse_args()

    print(f"Validating {'run ' + args.run if args.run else 'all data'}...")
    print("=" * 70)

    issues = validate(run_id=args.run)

    if not issues:
        print("✅ All checks passed")
        return 0

    by_severity = {"error": [], "warning": [], "info": []}
    for issue in issues:
        by_severity[issue.severity].append(issue)

    for severity in ("error", "warning", "info"):
        for issue in by_severity[severity]:
            print(issue)

    print("=" * 70)
    print(f"Summary: {len(by_severity['error'])} errors, "
          f"{len(by_severity['warning'])} warnings, "
          f"{len(by_severity['info'])} info")

    if args.strict and issues:
        return 1
    if args.errors_only and by_severity["error"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
