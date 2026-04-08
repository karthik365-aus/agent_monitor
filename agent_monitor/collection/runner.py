from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import uuid
from typing import Callable

from tqdm.auto import tqdm

from ..agents.base import Agent, AgentResult
from ..agents.registry import AgentRegistry
from ..config import SETTINGS
from ..metrics.factory import build_engine
from ..storage.store import store_result
from .query_pool import QueryPool


@dataclass
class CollectionPlan:
    """What to run. Built once, executed by collect()."""

    registry: AgentRegistry
    items: list[tuple[Agent, str]]
    run_id: str
    label: str | None = None

    def __len__(self) -> int:
        return len(self.items)


def build_plan(
    registry: AgentRegistry,
    pool: QueryPool,
    *,
    n_per_agent: int = 10,
    agents: list[str] | None = None,
    roles: list[str] | None = None,
    category: str | None = None,
    seed: int | None = None,
    shared_queries: bool = False,
    label: str | None = None,
    roles_to_exclude: tuple[str, ...] = ("judge",),
) -> CollectionPlan:
    selected = registry.all()
    selected = [a for a in selected if getattr(a, "enabled_for_collection", True)]
    if roles_to_exclude:
        selected = [a for a in selected if a.role not in roles_to_exclude]
    if agents:
        selected = [a for a in selected if a.name in agents]
    if roles:
        selected = [a for a in selected if a.role in roles]
    if not selected:
        raise ValueError("No agents matched filters")

    if shared_queries:
        cat = category or "support"
        queries = pool.sample(n_per_agent, category=cat, seed=seed)
        items = [(agent, query) for agent in selected for query in queries]
    else:
        items: list[tuple[Agent, str]] = []
        for agent in selected:
            cat = category or pool.categories_for(agent.role)
            for query in pool.sample(n_per_agent, category=cat, seed=seed):
                items.append((agent, query))

    return CollectionPlan(
        registry=registry,
        items=items,
        run_id=f"run_{uuid.uuid4().hex[:8]}",
        label=label,
    )


def collect(
    plan: CollectionPlan,
    *,
    parallel: int = 4,
    on_result: Callable[[AgentResult], None] | None = None,
    progress: bool = True,
) -> str:
    if not SETTINGS.agents_config.exists():
        raise FileNotFoundError(f"Missing agents config: {SETTINGS.agents_config}")
    primary = (SETTINGS.metrics or {}).get("primary")
    if primary == "keyword_hallucination":
        # Keep running, but surface that scoring quality is degraded.
        print("⚠️ primary scorer is keyword_hallucination; use llm_judge_hallucination for better quality.")

    engine = build_engine(plan.registry, SETTINGS.metrics or {})

    def _do(item: tuple[Agent, str]) -> AgentResult:
        agent, query = item
        result = agent.run(query)
        score = engine.score(result) if result.status == "success" else None
        store_result(result, run_id=plan.run_id, score=score)
        if on_result:
            on_result(result)
        return result

    bar = tqdm(
        total=len(plan),
        disable=not progress,
        desc=f"collecting {plan.label or plan.run_id}",
    )

    if parallel <= 1:
        for item in plan.items:
            _do(item)
            bar.update(1)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = [executor.submit(_do, item) for item in plan.items]
            for _ in as_completed(futures):
                bar.update(1)

    bar.close()
    return plan.run_id
