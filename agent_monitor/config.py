from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
import os
import yaml

logger = logging.getLogger(__name__)


def _infer_package_root() -> Path:
    """Directory containing `config/settings.yaml` (works for src layout and flat packages)."""
    here = Path(__file__).resolve().parent
    for base in (here, here.parent, here.parent.parent):
        if (base / "config" / "settings.yaml").is_file():
            return base
    # Fallback: nested package layout agent_monitor/agent_monitor/config.py
    return here.parent


# Project folder that contains `config/`, `pyproject.toml`, and `data/` (not cwd-dependent).
_PACKAGE_ROOT = _infer_package_root()
PACKAGE_ROOT = _PACKAGE_ROOT


def _resolve_config_path(value: str | Path) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (_PACKAGE_ROOT / p).resolve()


def _load_dotenv_if_present() -> None:
    """Lightweight .env loader so CLI works without manual export."""
    candidates = [
        Path.cwd() / ".env",
        _PACKAGE_ROOT / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    dotenv_path = next((p for p in candidates if p.exists()), None)
    if dotenv_path is None:
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row or row.startswith("#") or "=" not in row:
            continue
        key, value = row.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv_if_present()


@dataclass(frozen=True)
class Thresholds:
    failure_hallucination: float = 40
    warning_hallucination: float = 25
    success_rate_target: float = 90
    latency_warning: float = 2.0
    latency_critical: float = 3.0
    sample_size_min: int = 5
    p_value_threshold: float = 0.05
    drift_std_threshold: float = 2
    drift_latency_abs_threshold: float = 100.0
    forecast_days: int = 7


@dataclass(frozen=True)
class LeaderboardWeights:
    accuracy: float = 0.4
    speed: float = 0.3
    consistency: float = 0.3

    def __post_init__(self) -> None:
        total = self.accuracy + self.speed + self.consistency
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"weights.leaderboard must sum to 1.0, got {total}")


@dataclass(frozen=True)
class RoutingWeights:
    hallucination: float = 0.7
    latency: float = 0.3

    def __post_init__(self) -> None:
        total = self.hallucination + self.latency
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"weights.routing must sum to 1.0, got {total}")


@dataclass(frozen=True)
class RootCauseSeverityWeights:
    hallucination: float = 0.5
    latency: float = 0.3
    errors: float = 0.2

    def __post_init__(self) -> None:
        total = self.hallucination + self.latency + self.errors
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"weights.root_cause.severity must sum to 1.0, got {total}")


@dataclass(frozen=True)
class RootCauseRiskWeights:
    hallucination: float = 0.7
    latency: float = 0.2
    error_penalty: float = 30.0


@dataclass(frozen=True)
class RootCauseWeights:
    severity: RootCauseSeverityWeights = RootCauseSeverityWeights()
    risk: RootCauseRiskWeights = RootCauseRiskWeights()


@dataclass(frozen=True)
class Weights:
    leaderboard: LeaderboardWeights = LeaderboardWeights()
    routing: RoutingWeights = RoutingWeights()
    root_cause: RootCauseWeights = RootCauseWeights()
    latency_score_scale: float = 20.0


@dataclass(frozen=True)
class Settings:
    thresholds: Thresholds
    weights: Weights
    db_path: Path
    output_dir: Path
    agents_config: Path
    queries_config: Path
    prices_config: Path
    metrics: dict


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    settings_path = Path(path)
    if not settings_path.is_absolute():
        settings_path = (_PACKAGE_ROOT / settings_path).resolve()
    if not settings_path.exists():
        logger.warning(
            "Settings file not found at %s — using defaults (metrics/scorers may be empty). "
            "CWD-independent path is anchored to PACKAGE_ROOT=%s",
            settings_path,
            _PACKAGE_ROOT,
        )
    raw = yaml.safe_load(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
    weights_raw = raw.get("weights", {})
    root_cause_raw = weights_raw.get("root_cause", {})
    return Settings(
        thresholds=Thresholds(**raw.get("thresholds", {})),
        weights=Weights(
            leaderboard=LeaderboardWeights(**weights_raw.get("leaderboard", {})),
            routing=RoutingWeights(**weights_raw.get("routing", {})),
            root_cause=RootCauseWeights(
                severity=RootCauseSeverityWeights(**root_cause_raw.get("severity", {})),
                risk=RootCauseRiskWeights(**root_cause_raw.get("risk", {})),
            ),
            latency_score_scale=float(weights_raw.get("latency_score_scale", 20.0)),
        ),
        db_path=_resolve_config_path(os.environ.get("METRICS_DB", raw.get("db_path", "data/metrics.db"))),
        output_dir=_resolve_config_path(raw.get("output_dir", "data/reports")),
        agents_config=_resolve_config_path(raw.get("agents_config", "config/agents.yaml")),
        queries_config=_resolve_config_path(raw.get("queries_config", "config/queries.yaml")),
        prices_config=_resolve_config_path(raw.get("prices_config", "config/prices.yaml")),
        metrics=raw.get("metrics", {}),
    )


SETTINGS = load_settings()
