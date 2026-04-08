from __future__ import annotations

import os
import yaml

from .demo_wrapper import DemoNoiseWrapper
from .providers import GroqAgent, GeminiAgent

PROVIDERS = {
    "groq": GroqAgent,
    "gemini": GeminiAgent,
}


class AgentRegistry:
    def __init__(self):
        self._agents: dict[str, object] = {}

    @classmethod
    def from_yaml(cls, path: str) -> "AgentRegistry":
        cfg = yaml.safe_load(open(path, "r", encoding="utf-8"))
        defaults = cfg.get("defaults", {})
        reg = cls()
        for spec in cfg.get("agents", []):
            if not spec.get("enabled", True):
                continue
            merged = {**defaults, **spec}
            provider = merged.pop("provider")
            merged.pop("enabled", None)
            provider_cls = PROVIDERS[provider]
            agent = provider_cls(**merged)
            if os.environ.get("DEMO_MODE") == "1":
                agent = DemoNoiseWrapper(agent)
            reg._agents[merged["name"]] = agent
        return reg

    def get(self, name):
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' not found. Available: {list(self._agents.keys())}")
        return self._agents[name]

    def all(self):
        return list(self._agents.values())

    def by_role(self, role):
        return [a for a in self._agents.values() if a.role == role]

    def names(self):
        return list(self._agents)
