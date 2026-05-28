# ChinaTravel Agent Environment

This directory wraps the existing ChinaTravel environment for agent runtimes without
modifying the benchmark package. Official running and evaluation scripts such as
`run_exp.py`, `eval_exp.py`, and `eval_tpc.py` remain the source of truth.

## What This Provides

- `agent_env.adapter.ChinaTravelEnvAdapter`: Python API that lazily loads the official
  `WorldEnv` and returns JSON-serializable results.
- `agent_env.cli`: dependency-free command-line interface for one-shot calls and
  interactive terminal use.
- `agent_env.mcp_stdio`: dependency-free stdio MCP-style bridge for Codex-like clients.
- `agent_env.http_server`: dependency-free local HTTP JSON service for other agents or
  scripts.
- `agent_env/SKILL.md`: Codex skill instructions for using the CLI to solve benchmark
  queries.
- `agent_env/scripts/solve_one_with_codex.py`: one-query harness that loads a query,
  calls `codex exec`, saves the plan, and evaluates it.

## Prerequisites

Install the original project requirements and download the official database as usual:

```bash
pip install -r requirements.txt
# unzip the database to chinatravel/environment/database/
```

The wrapper itself can start without those dependencies, but environment tool calls will
return initialization errors until the official prerequisites are present.

## Codex/MCP-Style Usage

Configure the agent client to run:

```bash
python -m agent_env.mcp_stdio
```

The server exposes tools such as:

- `attractions_keys`
- `attractions_select`
- `restaurants_nearby`
- `goto`
- `intercity_transport_select`
- `poi_lat_lon_search`
- `china_travel_load_query`
- `china_travel_world_command`

Prefer the structured tools first. Use `china_travel_world_command` only when an
advanced query needs the original `WorldEnv` command-string surface.

## CLI Usage

List available tools:

```bash
python -m agent_env.cli tools
```

Call a structured tool:

```bash
python -m agent_env.cli call attractions_keys '{"city":"上海"}'
```

Call the original command-string interface:

```bash
python -m agent_env.cli world "attractions_keys('上海')"
```

Start an interactive prompt:

```bash
python -m agent_env.cli
```

Inside the prompt:

```text
tools
splits
call attractions_nearby {"city":"上海","point":"上海迪士尼度假区","topk":5,"dist":5}
world attractions_keys('上海')
quit
```

## One-Query Codex Harness

Run Codex non-interactively on one query and evaluate the output:

```bash
python agent_env/scripts/solve_one_with_codex.py --split easy
```

Use a specific query or model:

```bash
python agent_env/scripts/solve_one_with_codex.py --split easy --uid <uid>
python agent_env/scripts/solve_one_with_codex.py --split easy --codex-model gpt-5
```

The harness writes:

- prompt and raw Codex logs under `agent_env/runs/<split>_<uid>/`
- the itinerary under `results/codex_cli/<uid>.json`
- the one-query evaluation under `agent_env/runs/<split>_<uid>/evaluation.json`

It loads oracle fields internally for judging, but removes them from the query shown to
Codex. The nested `codex exec` workspace is also set to the run directory, with the
project root added via `--add-dir`, so Codex session metadata stays under the specific
run instead of the repository root.

## HTTP Usage

Start the local server:

```bash
python -m agent_env.http_server --host 127.0.0.1 --port 8765
```

List tools:

```bash
curl http://127.0.0.1:8765/tools
```

Call a tool:

```bash
curl -X POST http://127.0.0.1:8765/call \
  -H 'Content-Type: application/json' \
  -d '{"tool":"attractions_keys","arguments":{"city":"上海"}}'
```

Call the original command-string interface:

```bash
curl -X POST http://127.0.0.1:8765/world-command \
  -H 'Content-Type: application/json' \
  -d '{"command":"attractions_keys(\"上海\")"}'
```

## Boundary

This wrapper is for environment access and information lookup. It does not replace the
official agent execution or evaluation pipeline. Use the original scripts for benchmark
runs and scoring.
