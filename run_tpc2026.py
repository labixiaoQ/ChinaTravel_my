"""Unified TPC@IJCAI 2026 runner for bundled TPC2025 agents."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from func_timeout import FunctionTimedOut, func_timeout

from chinatravel.agent.load_model import init_agent, init_llm
from chinatravel.agent.registry import (
    AGENT_REGISTRY,
    agent_names,
    get_agent_spec,
    run_registered_agent,
)
from chinatravel.data.load_datasets import load_query, save_json_file
from chinatravel.environment.world_env import WorldEnv


PROJECT_ROOT = Path(__file__).resolve().parent


def method_name(args, llm_name):
    parts = [args.agent, llm_name]
    if args.lang == "en":
        parts.append("en")
    if args.agent == "LLM-modulo":
        parts.append("{}steps".format(args.refine_steps))
    if args.oracle_translation:
        parts.append("oracletranslation")
    if args.preference_search:
        parts.append("preferencesearch")
    return "_".join(parts)


def build_agent(args, log_dir):
    spec = get_agent_spec(args.agent)
    if spec.requires_hard_logic and not args.oracle_translation:
        raise ValueError("{} requires --oracle-translation".format(spec.name))

    llm_name = args.llm or spec.default_llm
    kwargs = {
        "method": spec.name,
        "env": WorldEnv(lang=args.lang),
        "backbone_llm": init_llm(llm_name, max_model_len=spec.max_model_len),
        "cache_dir": str(PROJECT_ROOT / "cache"),
        "log_dir": str(log_dir),
        "debug": args.debug,
        "refine_steps": args.refine_steps,
        "lang": args.lang,
    }
    return spec, llm_name, init_agent(kwargs)


def select_query_ids(query_ids, index=None, limit=None):
    if index is not None:
        if index not in query_ids:
            raise ValueError("Query uid not found in split: {}".format(index))
        return [index]
    if limit is not None:
        return query_ids[:limit]
    return query_ids


def run(args):
    query_ids, query_data = load_query(args)
    query_ids = select_query_ids(query_ids, args.index, args.limit)

    provisional_llm = args.llm or get_agent_spec(args.agent).default_llm
    method = method_name(args, provisional_llm)
    result_dir = PROJECT_ROOT / "results" / method
    log_dir = PROJECT_ROOT / "cache" / method
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    spec, llm_name, agent = build_agent(args, log_dir)
    print("agent={} llm={} lang={} split={} queries={}".format(
        spec.name, llm_name, args.lang, args.splits, len(query_ids)
    ))
    print("results={}".format(result_dir))

    success_count = 0
    completed_count = 0
    for position, uid in enumerate(query_ids, start=1):
        result_file = result_dir / "{}.json".format(uid)
        if args.skip and result_file.exists():
            print("[{}/{}] skip {}".format(position, len(query_ids), uid))
            continue

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        print("[{}/{}] run {}".format(position, len(query_ids), uid))
        try:
            success, plan = func_timeout(
                args.timeout,
                run_registered_agent,
                args=(agent, spec, query_data[uid], uid),
                kwargs={
                    "include_hard_logic": args.oracle_translation,
                    "preference_search": args.preference_search,
                },
            )
        except FunctionTimedOut:
            success, plan = False, {"error": "timeout after {}s".format(args.timeout)}
        except Exception as exc:
            success, plan = False, {
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

        save_json_file(plan, str(result_file))
        completed_count += 1
        success_count += int(bool(success))
        print("[{}/{}] {} {}".format(
            position, len(query_ids), "success" if success else "failed", uid
        ))

    print("completed={} success={}".format(completed_count, success_count))
    return 0 if completed_count == 0 or success_count == completed_count else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", "-s", default="TPC_IJCAI_2026_phase1")
    parser.add_argument("--index", "-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip", action="store_true")
    parser.add_argument("--agent", "-a", choices=agent_names(), default="UrbanTrip")
    parser.add_argument("--llm", "-l", default=None)
    parser.add_argument("--timeout", "-t", type=int, default=300)
    parser.add_argument("--lang", "--locale", choices=["zh", "en"], default="en")
    parser.add_argument(
        "--oracle-translation", "--oracle_translation",
        dest="oracle_translation", action=argparse.BooleanOptionalAction, default=True,
        help="Expose hard_logic_py to the agent (default: enabled).",
    )
    parser.add_argument("--preference-search", action="store_true")
    parser.add_argument("--refine-steps", type=int, default=10)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--list-agents", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def main():
    args = parse_args()
    if args.list_agents:
        for spec in AGENT_REGISTRY.values():
            print("{:<10} default_llm={:<9} {}".format(
                spec.name, spec.default_llm, spec.description
            ))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
