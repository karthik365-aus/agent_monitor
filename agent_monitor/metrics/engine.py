from __future__ import annotations

from dataclasses import dataclass, field

from .scorers import Score, Scorer
from ..agents.base import AgentResult


@dataclass
class ScoringResult:
    primary_hallucination: float
    confidence: float
    all_scores: dict[str, Score] = field(default_factory=dict)


class MetricsEngine:
    def __init__(self, scorers: list[Scorer], primary: str = "keyword_hallucination"):
        self.scorers = {scorer.name: scorer for scorer in scorers}
        self.primary = primary
        if primary not in self.scorers:
            raise ValueError(f"primary scorer '{primary}' not in {list(self.scorers)}")

    def score(self, result: AgentResult, **ctx) -> ScoringResult:
        scores: dict[str, Score] = {}
        for name, scorer in self.scorers.items():
            try:
                scores[name] = scorer.score(result.query, result.response, **ctx)
            except Exception as exc:  # noqa: BLE001
                scores[name] = Score(name=name, value=50.0, confidence=0.0, rationale=f"error: {exc}")
        primary = scores[self.primary]
        return ScoringResult(
            primary_hallucination=primary.value,
            confidence=primary.confidence,
            all_scores=scores,
        )
