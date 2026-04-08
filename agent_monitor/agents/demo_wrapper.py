from __future__ import annotations

import os
import random

DEMO_INJECTIONS = {
    "support": (0.20, " We also accept crypto!"),
    "summarization": (0.15, " This was written by Einstein!"),
    "analysis": (0.18, " Revenue will 10x next quarter!"),
}


class DemoNoiseWrapper:
    """Wraps any agent to inject demo hallucinations when DEMO_MODE=1."""

    def __init__(self, inner):
        self.__dict__.update(inner.__dict__)
        self._inner = inner
        self.name = inner.name

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def run(self, query: str):
        result = self._inner.run(query)
        if os.environ.get("DEMO_MODE") == "1" and result.status == "success":
            prob, suffix = DEMO_INJECTIONS.get(getattr(self._inner, "role", ""), (0, ""))
            if suffix and random.random() < prob:
                result.response += suffix
                result.metadata["demo_injected"] = True
        return result
