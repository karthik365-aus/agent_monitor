from __future__ import annotations


def latency_to_speed_score(latency_seconds: float, scale: float = 20.0) -> float:
    """Convert latency (seconds) to a 0-100 speed score."""
    return max(0.0, 100.0 - float(latency_seconds) * float(scale))


def latency_to_penalty(latency_seconds: float, scale: float = 20.0) -> float:
    """Convert latency to penalty points (0-100, lower latency is lower penalty)."""
    return max(0.0, 100.0 - latency_to_speed_score(latency_seconds, scale))
