from __future__ import annotations

from .engine import MetricsEngine
from .scorers import (
    Score,
    Scorer,
    EmbeddingRelevanceScorer,
    HeuristicQualityScorer,
    KeywordHallucinationScorer,
    LLMJudgeScorer,
)


class _UnavailableScorer(Scorer):
    def __init__(self, name: str, reason: str):
        self.name = name
        self.reason = reason

    def score(self, query: str, response: str, **ctx) -> Score:
        return Score(name=self.name, value=50.0, confidence=0.0, rationale=self.reason)


def build_engine(registry, cfg: dict) -> MetricsEngine:
    scorers = []
    scorer_names = cfg.get("scorers", ["keyword_hallucination"])

    for name in scorer_names:
        if name == "keyword_hallucination":
            scorers.append(KeywordHallucinationScorer())
        elif name == "llm_judge_hallucination":
            judge_name = cfg.get("judge_agent")
            if not judge_name:
                raise ValueError("'judge_agent' must be set in scorer config when using llm_judge_hallucination")
            judge = registry.get(judge_name)
            scorers.append(LLMJudgeScorer(judge))
        elif name == "semantic_relevance":
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(cfg.get("embedder", "all-MiniLM-L6-v2"))
                scorers.append(EmbeddingRelevanceScorer(lambda s: model.encode(s)))
            except Exception as exc:  # noqa: BLE001
                scorers.append(_UnavailableScorer("semantic_relevance", f"unavailable: {exc}"))
        elif name == "quality":
            scorers.append(HeuristicQualityScorer())
        else:
            raise ValueError(f"unknown scorer: {name}")

    primary = cfg.get("primary", scorer_names[0] if scorer_names else "keyword_hallucination")
    return MetricsEngine(scorers=scorers, primary=primary)
