from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
import logging
import random
from typing import Optional
import time
import yaml

from ..config import SETTINGS

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _load_prices() -> dict:
    try:
        raw = yaml.safe_load(SETTINGS.prices_config.read_text(encoding="utf-8")) or {}
        return raw.get("prices", {})
    except Exception as exc:
        logger.warning(f"could not load prices from {SETTINGS.prices_config}: {exc}")
        return {}


_NON_RETRYABLE_MARKERS = (
    "401",
    "403",
    "unauthorized",
    "forbidden",
    "invalid api key",
    "invalid_api_key",
    "api key not valid",
    "400",
    "bad request",
    "404",
    "not found",
    "model not found",
    "blocked",
    "safety",
    "permission denied",
    "quota exceeded",
)


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return not any(marker in msg for marker in _NON_RETRYABLE_MARKERS)


@dataclass
class AgentResult:
    agent: str
    query: str
    response: str
    latency: float
    model: str
    status: str = "success"
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    metadata: dict = field(default_factory=dict)


class Agent(ABC):
    def __init__(self, name: str, model: str, prompt_template: str, **kwargs):
        self.name = name
        self.model = model
        self.prompt_template = prompt_template
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 150)
        self.timeout = kwargs.get("timeout", 30)
        self.retries = int(kwargs.get("retries", 2))
        self.rate_limit_delay = float(kwargs.get("rate_limit_delay", 0.0))
        self.role = kwargs.get("role", "general")
        self.system_prompt = kwargs.get("system_prompt")
        self.enabled_for_collection = bool(kwargs.get("enabled_for_collection", True))
        self.backoff_base = float(kwargs.get("backoff_base", 1.0))

    def run(self, query: str) -> AgentResult:
        last_error: Optional[Exception] = None
        started = time.time()
        max_attempts = self.retries + 1
        attempts_used = 0

        for attempt in range(max_attempts):
            try:
                if self.rate_limit_delay > 0:
                    time.sleep(self.rate_limit_delay)

                prompt = self.prompt_template.format(query=query)
                text, usage = self._call(prompt)
                return AgentResult(
                    agent=self.name,
                    query=query,
                    response=text,
                    latency=time.time() - started,
                    model=self.model,
                    input_tokens=usage.get("in", 0),
                    output_tokens=usage.get("out", 0),
                    cost=self._price(usage),
                    metadata={"role": self.role, "attempts": attempt + 1},
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                attempts_used = attempt + 1
                exc_type = type(exc).__name__
                retryable = _is_retryable(exc)
                if not retryable:
                    logger.warning(
                        f"{self.name} attempt {attempt + 1}/{max_attempts} "
                        f"failed with non-retryable {exc_type}: {exc}"
                    )
                    break
                if attempt < max_attempts - 1:
                    backoff = self.backoff_base * (2**attempt)
                    jitter = random.uniform(0, self.backoff_base * 0.3)
                    sleep_for = backoff + jitter
                    logger.info(
                        f"{self.name} attempt {attempt + 1}/{max_attempts} failed with "
                        f"{exc_type}: {str(exc)[:120]} - retrying in {sleep_for:.1f}s"
                    )
                    time.sleep(sleep_for)
                else:
                    logger.warning(
                        f"{self.name} attempt {attempt + 1}/{max_attempts} "
                        f"failed with {exc_type}: {str(exc)[:120]} - giving up"
                    )

        return AgentResult(
            agent=self.name,
            query=query,
            response="",
            latency=time.time() - started,
            model=self.model,
            status="error",
            error=f"{type(last_error).__name__}: {str(last_error)[:200]}" if last_error else "unknown error",
            metadata={"role": self.role, "attempts": attempts_used},
        )

    @abstractmethod
    def _call(self, prompt: str) -> tuple[str, dict]:
        raise NotImplementedError

    def _price(self, usage: dict) -> float:
        price_map = _load_prices()
        p = price_map.get(self.model, {"in": 0.0, "out": 0.0})
        return (usage.get("in", 0) * p["in"] + usage.get("out", 0) * p["out"]) / 1_000_000
