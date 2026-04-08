from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import numpy as np
import re


@dataclass
class Score:
    name: str
    value: float
    confidence: float = 1.0
    rationale: str = ""


class Scorer(ABC):
    name: str

    @abstractmethod
    def score(self, query: str, response: str, **ctx) -> Score:
        raise NotImplementedError


class KeywordHallucinationScorer(Scorer):
    name = "keyword_hallucination"
    SUSPICIOUS = [
        r"\b10x\b",
        r"\bguaranteed\b",
        r"\bdefinitely\b",
        r"\bnobel\b",
        r"\beinstein\b",
        r"\bcrypto\b",
        r"\bwill\s+(?:10x|moon|skyrocket)\b",
    ]

    def score(self, query: str, response: str, **ctx) -> Score:
        hits = sum(1 for pattern in self.SUSPICIOUS if re.search(pattern, response or "", re.I))
        value = min(90.0, 10.0 + hits * 16.0)
        return Score(self.name, value, confidence=0.3, rationale=f"{hits} suspicious patterns matched")


class LLMJudgeScorer(Scorer):
    name = "llm_judge_hallucination"
    PROMPT = """You are a strict factuality grader. Rate RESPONSE hallucination on 0-100.
Use this rubric:
  0   = every claim is directly verifiable from QUERY (rare)
  20  = mostly grounded, but includes mild paraphrase not explicitly in QUERY
  50  = contains vague or unverifiable claims
  90+ = fabricated facts, fake quotes, or invented numbers

Return ONLY a JSON object: {{"score": <int>, "reason": "<one sentence>"}}

EXAMPLE A
QUERY: "Return policy?"
RESPONSE: "Returns are accepted within 30 days with receipt."
OUTPUT: {{"score": 0, "reason": "All claims are directly grounded in the query context."}}

EXAMPLE B
QUERY: "Shipping time?"
RESPONSE: "Shipping usually takes a few days depending on carrier and region."
OUTPUT: {{"score": 50, "reason": "The response is plausible but not verifiable from the query alone."}}

EXAMPLE C
QUERY: "Revenue trend?"
RESPONSE: "Revenue grew 42% last quarter and Gartner named us #1."
OUTPUT: {{"score": 90, "reason": "The response introduces specific unsupported facts and rankings."}}

QUERY: {query}
{reference_block}
RESPONSE: {response}"""

    def __init__(self, judge_agent):
        self.judge = judge_agent

    def score(self, query: str, response: str, **ctx) -> Score:
        if not (response or "").strip():
            return Score(self.name, 50.0, confidence=0.0, rationale="empty response")
        q_raw = query or ""
        r_raw = response or ""
        q_cut = q_raw[:500]
        r_cut = r_raw[:1500]
        reference = (ctx.get("reference") or ctx.get("reference_answer") or "").strip()
        reference_block = f'REFERENCE_ANSWER: "{reference[:1500]}"' if reference else ""
        was_q_trunc = len(q_raw) > len(q_cut)
        was_r_trunc = len(r_raw) > len(r_cut)
        prompt = self.PROMPT.format(query=q_cut, response=r_cut, reference_block=reference_block)
        result = self.judge.run(prompt)
        if result.status != "success":
            return Score(self.name, 50.0, confidence=0.0, rationale=f"judge failed: {result.error}")
        try:
            text = re.sub(r"```(?:json)?|```", "", result.response).strip()
            data = json.loads(text)
            reason = str(data.get("reason", ""))[:200]
            if was_q_trunc or was_r_trunc:
                reason = (reason + f" [truncated q={len(q_raw)}->{len(q_cut)}, r={len(r_raw)}->{len(r_cut)}]")[:200]
            return Score(
                self.name,
                float(data["score"]),
                confidence=0.85,
                rationale=reason,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            return Score(self.name, 50.0, confidence=0.0, rationale=f"parse error: {exc}")


class EmbeddingRelevanceScorer(Scorer):
    name = "semantic_relevance"

    def __init__(self, embedder):
        self.embed = embedder

    def score(self, query: str, response: str, **ctx) -> Score:
        if not (query or "").strip() or not (response or "").strip():
            return Score(self.name, 0.0, confidence=0.0)
        q_vec = self.embed(query)
        r_vec = self.embed(response)
        cos = float(np.dot(q_vec, r_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(r_vec) + 1e-9))
        value = max(0.0, min(100.0, (cos + 1) * 50.0))
        return Score(self.name, value, confidence=0.9)


class HeuristicQualityScorer(Scorer):
    name = "quality"

    def score(self, query: str, response: str, **ctx) -> Score:
        if not response:
            return Score(self.name, 0.0, confidence=0.5)
        words = len(response.split())
        value = 50.0
        if 10 <= words <= 100:
            value += 20
        elif 5 <= words < 10 or 100 < words <= 150:
            value += 10
        if response.count(".") >= 2:
            value += 15
        if any(word in response.lower() for word in ("specifically", "example", "analysis")):
            value += 10
        return Score(self.name, min(100.0, value), confidence=0.5)
