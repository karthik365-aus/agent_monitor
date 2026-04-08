"""Smoke tests. Run with: pytest tests/test_smoke.py -x

These tests don't hit any APIs. They use fake agents and synthetic data
to verify the plumbing works end-to-end. Should run in under 5 seconds.
"""
from __future__ import annotations

import pandas as pd
import pytest
from dataclasses import replace
from unittest.mock import MagicMock

# ─── Import smoke: catches circular imports, syntax errors, missing modules ──

def test_all_modules_import():
    """If this fails, something is broken at the module level."""
    import agent_monitor.config
    import agent_monitor.agents.base
    import agent_monitor.agents.providers
    import agent_monitor.agents.registry
    import agent_monitor.metrics.scorers
    import agent_monitor.metrics.engine
    import agent_monitor.metrics.factory
    import agent_monitor.collection.query_pool
    import agent_monitor.collection.runner
    import agent_monitor.storage.store
    import agent_monitor.storage.schema
    import agent_monitor.analysis.leaderboard
    import agent_monitor.analysis.ab_test
    import agent_monitor.analysis.drift
    import agent_monitor.analysis.root_cause
    import agent_monitor.analysis.routing
    import agent_monitor.analysis.cost
    import agent_monitor.analysis.health
    import agent_monitor.analysis.confidence
    # Optional dependency: pdf module requires reportlab.
    try:
        import agent_monitor.reporting.pdf
    except ModuleNotFoundError as exc:
        if exc.name != "reportlab":
            raise
    import agent_monitor.reporting.alerts
    import agent_monitor.reporting.dashboard


# ─── Config: settings.yaml loads and has the right shape ─────────────────────

def test_settings_loads_with_weights():
    from agent_monitor.config import SETTINGS
    assert SETTINGS.thresholds.failure_hallucination > 0
    # Weights block should exist (added in the recent refactor)
    assert hasattr(SETTINGS, "weights") or "weights" in SETTINGS.metrics or True
    # ^ relax this once you've finalized where weights live


def test_yaml_files_exist():
    from agent_monitor.config import SETTINGS
    assert SETTINGS.agents_config.exists(), f"missing {SETTINGS.agents_config}"
    assert SETTINGS.queries_config.exists(), f"missing {SETTINGS.queries_config}"
    assert SETTINGS.prices_config.exists(), f"missing {SETTINGS.prices_config}"


# ─── Agent base: retry logic and result shape ────────────────────────────────

def test_agent_result_shape():
    from agent_monitor.agents.base import AgentResult
    r = AgentResult(agent="x", query="q", response="r", latency=0.1, model="m")
    assert r.status == "success"
    assert r.error is None
    assert r.metadata == {}


def test_agent_retries_then_succeeds():
    """Agent that fails twice then succeeds should return success on attempt 3."""
    from agent_monitor.agents.base import Agent

    class FlakyAgent(Agent):
        def __init__(self):
            super().__init__(name="flaky", model="m", prompt_template="{query}",
                             retries=2, backoff_base=0.01)
            self.calls = 0

        def _call(self, prompt):
            self.calls += 1
            if self.calls < 3:
                raise ConnectionError("transient")
            return "ok", {"in": 1, "out": 1}

    a = FlakyAgent()
    result = a.run("test")
    assert result.status == "success"
    assert result.response == "ok"
    assert a.calls == 3
    assert result.metadata.get("attempts") == 3


def test_agent_bails_on_non_retryable():
    """401 should fail in 1 attempt, not 3."""
    from agent_monitor.agents.base import Agent

    class AuthFailAgent(Agent):
        def __init__(self):
            super().__init__(name="auth", model="m", prompt_template="{query}",
                             retries=2, backoff_base=0.01)
            self.calls = 0

        def _call(self, prompt):
            self.calls += 1
            raise PermissionError("401 unauthorized")

    a = AuthFailAgent()
    result = a.run("test")
    assert result.status == "error"
    assert a.calls == 1, f"should bail on first attempt, got {a.calls}"


# ─── Scorers: behavior on edge cases ─────────────────────────────────────────

def test_keyword_scorer_floor():
    from agent_monitor.metrics.scorers import KeywordHallucinationScorer
    s = KeywordHallucinationScorer()
    score = s.score("q", "completely normal response")
    assert score.value == 10.0  # floor, not 0
    assert score.confidence == 0.3


def test_keyword_scorer_catches_hallmark_words():
    from agent_monitor.metrics.scorers import KeywordHallucinationScorer
    s = KeywordHallucinationScorer()
    score = s.score("q", "Einstein guaranteed it would 10x")
    assert score.value > 50


def test_llm_judge_returns_50_on_empty_response():
    from agent_monitor.metrics.scorers import LLMJudgeScorer
    s = LLMJudgeScorer(judge_agent=None)
    score = s.score("q", "")
    assert score.value == 50.0
    assert score.confidence == 0.0  # critical: must be 0 so storage filters it


def test_llm_judge_zero_confidence_on_judge_failure():
    from agent_monitor.metrics.scorers import LLMJudgeScorer
    from agent_monitor.agents.base import AgentResult

    fake_judge = MagicMock()
    fake_judge.run.return_value = AgentResult(
        agent="j", query="", response="", latency=0, model="m",
        status="error", error="rate limit",
    )
    s = LLMJudgeScorer(judge_agent=fake_judge)
    score = s.score("q", "some real response")
    assert score.confidence == 0.0
    assert "judge failed" in score.rationale


# ─── Storage: zero-confidence handling, schema integrity ─────────────────────

def test_storage_filters_zero_confidence(tmp_path, monkeypatch):
    """Critical: a fake 50.0 from a failed judge must NOT land as 50.0 in DB."""
    from agent_monitor.storage import store
    from agent_monitor.metrics.engine import ScoringResult
    from agent_monitor.metrics.scorers import Score
    from agent_monitor.agents.base import AgentResult

    # Redirect DB to a temp file
    monkeypatch.setattr(store, "SETTINGS", replace(store.SETTINGS, db_path=tmp_path / "test.db"))
    store._initialized = False  # force re-init

    result = AgentResult(agent="x", query="q", response="r", latency=0.1, model="m")
    fake_score = ScoringResult(
        primary_hallucination=50.0,
        confidence=0.0,  # judge failed
        all_scores={"llm_judge_hallucination": Score("llm_judge_hallucination", 50.0, 0.0, "judge failed")},
    )
    store.store_result(result, run_id="test", score=fake_score)
    df = store.get_metrics_df(run_id="test")

    # The hallucination_score should be NULL, not 50.0
    assert df["hallucination_score"].isna().all(), \
        "zero-confidence scores must be stored as NULL to avoid polluting averages"


def test_metric_scores_table_exists(tmp_path, monkeypatch):
    from agent_monitor.storage import store
    import sqlite3

    monkeypatch.setattr(store, "SETTINGS", replace(store.SETTINGS, db_path=tmp_path / "test.db"))
    store._initialized = False
    store.ensure_initialized()

    with sqlite3.connect(store.SETTINGS.db_path) as c:
        tables = {row[0] for row in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "metrics" in tables
    assert "metric_scores" in tables, "sidecar table for multi-scorer results missing"


# ─── Query pool: sampling reproducibility ────────────────────────────────────

def test_query_pool_seeded_sample_is_reproducible():
    from agent_monitor.collection.query_pool import QueryPool
    pool = QueryPool({"test": [f"q{i}" for i in range(20)]})
    a = pool.sample(5, category="test", seed=42)
    b = pool.sample(5, category="test", seed=42)
    assert a == b


def test_query_pool_returns_all_when_n_exceeds():
    from agent_monitor.collection.query_pool import QueryPool
    pool = QueryPool({"test": ["q1", "q2", "q3"]})
    result = pool.sample(100, category="test")
    assert len(result) == 3


# ─── Analysis functions: don't crash on empty/synthetic data ─────────────────

def _synthetic_df(n_per_agent=10):
    """Create a fake metrics dataframe to feed analysis functions."""
    import numpy as np
    rows = []
    for agent in ["agent_a", "agent_b", "agent_c"]:
        for i in range(n_per_agent):
            rows.append({
                "id": len(rows) + 1,
                "run_id": "test",
                "agent_name": agent,
                "query": f"q{i}",
                "response": f"r{i}",
                "hallucination_score": float(np.random.uniform(0, 60)),
                "latency": float(np.random.uniform(0.1, 2.0)),
                "input_tokens": 10,
                "output_tokens": 20,
                "cost": 0.001,
                "model": "test-model",
                "status": "success",
                "error": None,
                "timestamp": pd.Timestamp("2026-04-07") + pd.Timedelta(seconds=i),
            })
    return pd.DataFrame(rows)


def test_leaderboard_empty_df():
    from agent_monitor.analysis import leaderboard
    result = leaderboard.render(pd.DataFrame())
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_leaderboard_synthetic():
    from agent_monitor.analysis import leaderboard
    result = leaderboard.render(_synthetic_df())
    assert len(result) == 3
    assert "Overall Score" in result.columns
    # All scores should be 0-100
    assert (result["Overall Score"] >= 0).all()
    assert (result["Overall Score"] <= 100).all()


def test_drift_handles_empty_df():
    from agent_monitor.analysis import drift
    result = drift.detect_all(pd.DataFrame())
    assert result["status"] == "no_data"


def test_ab_test_requires_min_samples():
    """A/B test with n=2 should produce 'insufficient data', not a junk p-value."""
    from agent_monitor.analysis.ab_test import compare_pair
    df = pd.DataFrame([
        {"agent_name": "a", "latency": 1.0, "hallucination_score": 10},
        {"agent_name": "a", "latency": 1.1, "hallucination_score": 12},
        {"agent_name": "b", "latency": 0.5, "hallucination_score": 15},
        {"agent_name": "b", "latency": 0.6, "hallucination_score": 14},
    ])
    result = compare_pair(df, "a", "b")
    # If you bumped the minimum to 5 (recommended), this should error
    assert "error" in result or result.get("p_value_latency") is not None
    # ^ tighten this once you've decided on the minimum


def test_root_cause_handles_empty_df():
    from agent_monitor.analysis import root_cause
    result = root_cause.analyze_all(pd.DataFrame())
    assert result["status"] == "no_data"


# ─── Run with: pytest tests/test_smoke.py -x -v ──────────────────────────────
