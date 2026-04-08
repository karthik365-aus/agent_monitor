from __future__ import annotations

import sqlite3
import threading

import pandas as pd

from ..config import SETTINGS
from ..agents.base import AgentResult
from .schema import DDL

_lock = threading.Lock()
_initialized = False


def _conn():
    global _initialized
    SETTINGS.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SETTINGS.db_path)
    if not _initialized:
        with _lock:
            if not _initialized:
                for stmt in DDL:
                    conn.execute(stmt)
                conn.commit()
                _initialized = True
    return conn


def ensure_initialized() -> None:
    with _conn() as _:
        pass


def store_result(r: AgentResult, run_id: str | None = None, score: float | object | None = None):
    if hasattr(score, "primary_hallucination"):
        primary_score = score.primary_hallucination
        detailed_scores = getattr(score, "all_scores", {})
        top_confidence = getattr(score, "confidence", None)
        if top_confidence == 0.0:
            primary_to_store = None
        else:
            primary_to_store = primary_score
    else:
        primary_score = score
        detailed_scores = {}
        top_confidence = None
        primary_to_store = primary_score

    with _conn() as c:
        cur = c.execute(
            """
            INSERT INTO metrics (
              run_id, agent_name, query, response, hallucination_score,
              latency, input_tokens, output_tokens, cost, model, status, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                r.agent,
                r.query,
                r.response,
                primary_to_store,
                r.latency,
                r.input_tokens,
                r.output_tokens,
                r.cost,
                r.model,
                r.status,
                r.error,
            ),
        )
        metric_id = cur.lastrowid

        for scorer_name, scorer_score in detailed_scores.items():
            c.execute(
                """
                INSERT INTO metric_scores (metric_id, scorer_name, value, confidence, rationale)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    metric_id,
                    scorer_name,
                    scorer_score.value,
                    scorer_score.confidence,
                    scorer_score.rationale,
                ),
            )


def get_metrics_df(run_id: str | None = None, agent: str | None = None) -> pd.DataFrame:
    query = "SELECT * FROM metrics WHERE 1=1"
    params: list[object] = []
    if run_id:
        query += " AND run_id=?"
        params.append(run_id)
    if agent:
        query += " AND agent_name=?"
        params.append(agent)
    with _conn() as c:
        return pd.read_sql_query(query, c, params=params)


def get_agent_stats(agent_name: str) -> dict:
    df = get_metrics_df(agent=agent_name)
    if df.empty:
        return {"agent": agent_name, "count": 0}
    return {
        "agent": agent_name,
        "count": int(len(df)),
        "avg_latency": float(df["latency"].mean()),
        "avg_hallucination": float(df["hallucination_score"].mean()),
    }


def get_metric_scores_df(run_id: str | None = None) -> pd.DataFrame:
    query = """
    SELECT s.*, m.run_id, m.agent_name, m.timestamp
    FROM metric_scores s
    JOIN metrics m ON m.id = s.metric_id
    WHERE 1=1
    """
    params: list[object] = []
    if run_id:
        query += " AND m.run_id=?"
        params.append(run_id)
    with _conn() as c:
        return pd.read_sql_query(query, c, params=params)
