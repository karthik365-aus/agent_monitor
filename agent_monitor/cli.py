from __future__ import annotations

import logging

import typer

from .agents.registry import AgentRegistry
from .collection.query_pool import QueryPool
from .collection.runner import build_plan, collect
from .config import SETTINGS
from .storage.store import get_metrics_df
from .analysis import leaderboard

app = typer.Typer()


def _configure_logging(verbose: bool) -> None:
    """So Agent.run retry/backoff logs are visible in the CLI (not swallowed by default)."""
    level = logging.DEBUG if verbose else logging.INFO
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    else:
        logging.getLogger().setLevel(level)


@app.callback()
def _main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging (includes retry details)."),
) -> None:
    _configure_logging(verbose)


@app.command()
def run(
    n: int = typer.Option(10, "-n", "--n-per-agent", help="Number of queries per agent."),
    agents: str = "",
    role: str = "",
    category: str = "",
    parallel: int = typer.Option(4, "--parallel", "-p", help="Concurrent workers for collection."),
    seed: int | None = None,
    shared: bool = False,
    label: str = "",
    dry_run: bool = False,
) -> None:
    reg = AgentRegistry.from_yaml(str(SETTINGS.agents_config))
    pool = QueryPool.from_yaml(str(SETTINGS.queries_config))
    plan = build_plan(
        reg,
        pool,
        n_per_agent=n,
        agents=[a for a in agents.split(",") if a] or None,
        roles=[role] if role else None,
        category=category or None,
        seed=seed,
        shared_queries=shared,
        label=label or None,
    )
    agent_count = len({a.name for a, _ in plan.items})
    typer.echo(f"Plan: {len(plan)} calls across {agent_count} agents")
    if dry_run:
        return
    run_id = collect(plan, parallel=parallel)
    typer.echo(f"✅ run_id={run_id}")


@app.command()
def analyze(run: str | None = None) -> None:
    df = get_metrics_df(run_id=run)
    typer.echo(leaderboard.render(df).to_string(index=False))


if __name__ == "__main__":
    app()
