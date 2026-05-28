#!/usr/bin/env python3
"""Solve one ChinaTravel query with non-interactive Codex and evaluate it."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HIDDEN_QUERY_KEYS = {"hard_logic", "hard_logic_py", "hard_logic_nl"}


def load_one_query(split: str, uid: str | None) -> tuple[str, dict[str, Any]]:
    from chinatravel.data.load_datasets import load_query

    args = argparse.Namespace(splits=split, oracle_translation=True)
    query_ids, query_data = load_query(args)
    if not query_ids:
        raise RuntimeError(f"No queries found for split: {split}")

    query_uid = uid or query_ids[0]
    if query_uid not in query_data:
        raise RuntimeError(f"Query uid {query_uid!r} not found in split {split!r}")
    return query_uid, query_data[query_uid]


def visible_query(query: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in query.items() if key not in HIDDEN_QUERY_KEYS}


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Codex output was valid JSON but not a JSON object.")
    return value


def build_prompt(split: str, uid: str, query: dict[str, Any]) -> str:
    query_text = json.dumps(query, ensure_ascii=False, indent=2)
    return f"""You are solving one ChinaTravel benchmark query.

Use the local CLI API to inspect the environment. The repository root is:
{PROJECT_ROOT}

Run CLI commands from that repository root, either by setting the shell command working
directory to that path or by prefixing commands with `cd {PROJECT_ROOT} &&`.

Important commands:
- python -m agent_env.cli tools
- python -m agent_env.cli call attractions_keys '{{"city":"<target_city>"}}'
- python -m agent_env.cli call restaurants_keys '{{"city":"<target_city>"}}'
- python -m agent_env.cli call accommodations_keys '{{"city":"<target_city>"}}'
- python -m agent_env.cli call intercity_transport_select '{{"start_city":"<start_city>","end_city":"<target_city>","intercity_type":"train","earliest_leave_time":"07:00"}}'
- python -m agent_env.cli call goto '{{"city":"<target_city>","start":"<exact start>","end":"<exact end>","start_time":"HH:MM","transport_type":"metro"}}'

Use exact names, prices, costs, times, distances, TrainID/FlightID values, and transport
segments returned by the CLI. Do not invent environment facts.

Return only a JSON object matching chinatravel/evaluation/output_schema.json.
Do not wrap the final answer in Markdown.

Split: {split}
UID: {uid}
Visible query:
{query_text}
"""


def run_codex(
    prompt: str,
    run_dir: Path,
    output_path: Path,
    model: str | None,
    timeout: int,
    sandbox: str,
) -> subprocess.CompletedProcess[str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "codex",
        "exec",
        "--cd",
        str(run_dir),
        "--add-dir",
        str(PROJECT_ROOT),
        "--sandbox",
        sandbox,
        "--output-last-message",
        str(output_path),
    ]
    if model:
        cmd.extend(["--model", model])
    cmd.append("-")

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.run(
        cmd,
        cwd=run_dir,
        env=env,
        text=True,
        input=prompt,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def evaluate_one(split: str, uid: str, query: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    from chinatravel.evaluation.commonsense_constraint import evaluate_commonsense_constraints
    from chinatravel.evaluation.hard_constraint import evaluate_hard_constraints_v2
    from chinatravel.evaluation.schema_constraint import evaluate_schema_constraints
    from chinatravel.evaluation.utils import load_json_file

    query_index = [uid]
    query_data = {uid: query}
    result_data = {uid: plan}
    schema = load_json_file(str(PROJECT_ROOT / "chinatravel" / "evaluation" / "output_schema.json"))

    schema_rate, schema_result_agg, schema_pass_id = evaluate_schema_constraints(
        query_index, result_data, schema=schema
    )
    macro_comm, micro_comm, common_result_agg, commonsense_pass_id = evaluate_commonsense_constraints(
        query_index, query_data, result_data, verbose=False
    )
    (
        macro_logi,
        micro_logi,
        conditional_macro_logi,
        conditional_micro_logi,
        logi_result_agg,
        logi_pass_id,
    ) = evaluate_hard_constraints_v2(
        query_index,
        query_data,
        result_data,
        env_pass_id=commonsense_pass_id,
        verbose=False,
    )

    return {
        "split": split,
        "uid": uid,
        "schema_pass": uid in schema_pass_id,
        "commonsense_pass": uid in commonsense_pass_id,
        "logical_pass": uid in logi_pass_id,
        "all_pass": uid in set(schema_pass_id) & set(commonsense_pass_id) & set(logi_pass_id),
        "schema_rate": schema_rate,
        "commonsense_micro": micro_comm,
        "commonsense_macro": macro_comm,
        "logical_micro": micro_logi,
        "logical_macro": macro_logi,
        "conditional_logical_micro": conditional_micro_logi,
        "conditional_logical_macro": conditional_macro_logi,
        "schema_details": schema_result_agg.to_dict(orient="records"),
        "commonsense_details": common_result_agg.to_dict(orient="records"),
        "logical_details": logi_result_agg.to_dict(orient="records"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load one ChinaTravel query, solve it with codex exec, and evaluate the output."
    )
    parser.add_argument("--split", default="easy", help="ChinaTravel split name.")
    parser.add_argument("--uid", default=None, help="Optional query UID. Defaults to first query in split.")
    parser.add_argument("--method", default="codex_cli", help="Result/eval method directory name.")
    parser.add_argument("--codex-model", default=None, help="Optional model passed to codex exec.")
    parser.add_argument("--timeout", type=int, default=900, help="Codex subprocess timeout in seconds.")
    parser.add_argument("--sandbox", default="workspace-write", choices=["read-only", "workspace-write", "danger-full-access"])
    parser.add_argument("--no-run-codex", action="store_true", help="Load the query and write the prompt, but do not call Codex.")
    parser.add_argument("--plan-file", default=None, help="Evaluate an existing plan JSON instead of calling Codex.")
    parser.add_argument("--work-dir", default="agent_env/runs", help="Directory for prompts, raw Codex output, and evaluation JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uid, query = load_one_query(args.split, args.uid)
    public_query = visible_query(query)

    run_dir = PROJECT_ROOT / args.work_dir / f"{args.split}_{uid}"
    prompt_path = run_dir / "prompt.txt"
    raw_output_path = run_dir / "codex_output.json"
    eval_path = run_dir / "evaluation.json"
    result_path = PROJECT_ROOT / "results" / args.method / f"{uid}.json"

    prompt = build_prompt(args.split, uid, public_query)
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")

    print(f"Loaded query: split={args.split} uid={uid}")
    print(f"Prompt: {prompt_path}")

    if args.plan_file:
        plan = read_json(Path(args.plan_file))
    elif args.no_run_codex:
        print("Skipping Codex call because --no-run-codex was set.")
        return
    else:
        completed = run_codex(
            prompt=prompt,
            run_dir=run_dir,
            output_path=raw_output_path,
            model=args.codex_model,
            timeout=args.timeout,
            sandbox=args.sandbox,
        )
        (run_dir / "codex_stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (run_dir / "codex_stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"codex exec failed with exit code {completed.returncode}. "
                f"See {run_dir / 'codex_stdout.txt'} and {run_dir / 'codex_stderr.txt'}."
            )
        plan = extract_json_object(raw_output_path.read_text(encoding="utf-8"))

    write_json(result_path, plan)
    print(f"Saved plan: {result_path}")

    evaluation = evaluate_one(args.split, uid, query, plan)
    write_json(eval_path, evaluation)
    print(f"Saved evaluation: {eval_path}")
    print(json.dumps(evaluation, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
