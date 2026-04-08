# agent_monitor

Modular AI agent performance monitor.

## Quick start

- Configure env vars from `.env.example`
- Configure agents in `config/agents.yaml`
- Configure settings in `config/settings.yaml`
- Run collection: `python -m agent_monitor.cli run -n 20 --label smoke`
- Run analysis: `python -m agent_monitor.cli analyze`
- Start dashboard: `streamlit run dashboard/app.py`

## Useful examples

- Fair A/B sample set: `python -m agent_monitor.cli run --role support --shared --seed 42 -n 30 --label prompt-ab`
- Dry-run plan preview: `python -m agent_monitor.cli run --agents groq_support --category edge_cases -n 20 --dry-run`
- Equivalent long flag for sample count: `python -m agent_monitor.cli run --n-per-agent 20 --dry-run`

## Quality checks

- Fast smoke checks after code changes: `pytest tests/test_smoke.py -x`
- Data quality checks after a collection run:
  - All data: `python validate.py --errors-only`
  - One run: `python validate.py --run <run_id> --errors-only`
- Strict CI-style gate: `pytest tests/test_smoke.py -x && python validate.py --errors-only`
