from __future__ import annotations

import random
from pathlib import Path
import yaml


class QueryPool:
    def __init__(self, categories: dict[str, list[str]]):
        self.categories = categories

    @classmethod
    def from_yaml(cls, path: str | Path) -> "QueryPool":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls(raw.get("categories", {}))

    def sample(
        self,
        n: int,
        category: str | None = None,
        seed: int | None = None,
    ) -> list[str]:
        rng = random.Random(seed)
        if category and category in self.categories:
            pool = self.categories[category]
        else:
            pool = [q for arr in self.categories.values() for q in arr]
        if not pool:
            return []
        if n >= len(pool):
            return list(pool)
        return rng.sample(pool, n)

    def all(self, category: str | None = None) -> list[str]:
        if category:
            return list(self.categories.get(category, []))
        return [q for qs in self.categories.values() for q in qs]

    def categories_for(self, role: str) -> str:
        selected = {
            "support": "support",
            "summarization": "summarization",
            "analysis": "analysis",
            "edge_case_test": "edge_cases",
        }.get(role, role)
        return selected
