DDL = [
    """
    CREATE TABLE IF NOT EXISTS metrics (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id TEXT,
      agent_name TEXT,
      query TEXT,
      response TEXT,
      hallucination_score FLOAT,
      latency FLOAT,
      input_tokens INTEGER,
      output_tokens INTEGER,
      cost FLOAT,
      model TEXT,
      status TEXT,
      error TEXT,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS metric_scores (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      metric_id INTEGER REFERENCES metrics(id) ON DELETE CASCADE,
      scorer_name TEXT NOT NULL,
      value REAL NOT NULL,
      confidence REAL NOT NULL,
      rationale TEXT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_metric_scores_metric
    ON metric_scores(metric_id)
    """,
    "CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON metrics(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_metrics_agent_name ON metrics(agent_name)",
    "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_metrics_status ON metrics(status)",
]
