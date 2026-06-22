"""Score only the result files currently produced for a TPC split.

This program reuses the scoring functions and formula from ``eval_tpc.py``.
The only difference is that the evaluated query list is restricted to JSON
files present in the selected results directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chinatravel.data.load_datasets import load_query
from chinatravel.evaluation.commonsense_constraint import (
    evaluate_commonsense_constraints,
)
from chinatravel.evaluation.hard_constraint import evaluate_hard_constraints_v2
from chinatravel.evaluation.preference import evaluate_preference_v2
from chinatravel.evaluation.schema_constraint import evaluate_schema_constraints
from chinatravel.evaluation.utils import load_json_file
from eval_tpc import _method_has_en_suffix, cal_default_pr_score, load_result


PROJECT_ROOT = Path(__file__).resolve().parent


def existing_result_ids(query_ids, results_dir):
    """Return existing result ids in the original split order."""

    results_dir = Path(results_dir)
    return [uid for uid in query_ids if (results_dir / (uid + ".json")).is_file()]


def evaluate_partial(args):
    if args.lang == "en" and not _method_has_en_suffix(args.method):
        args.method += "_en"

    all_query_ids, query_data = load_query(args)
    results_dir = Path(args.results_dir or PROJECT_ROOT / "results" / args.method)
    query_ids = existing_result_ids(all_query_ids, results_dir)
    if not query_ids:
        raise FileNotFoundError(
            "No result JSON files for split {!r} under {}".format(
                args.splits, results_dir
            )
        )

    print("Method: {}".format(args.method))
    print("Results: {}".format(results_dir))
    print(
        "Evaluating {}/{} outputs ({:.2f}% coverage)".format(
            len(query_ids),
            len(all_query_ids),
            100.0 * len(query_ids) / len(all_query_ids),
        )
    )

    method = "default"
    _, result_data_by_method = load_result(args, query_ids, str(results_dir))
    result_data = result_data_by_method[method]
    schema = load_json_file(
        str(PROJECT_ROOT / "chinatravel" / "evaluation" / "output_schema.json")
    )

    _, _, schema_pass_ids = evaluate_schema_constraints(
        query_ids, result_data, schema=schema
    )
    macro_comm, micro_comm, _, commonsense_pass_ids = (
        evaluate_commonsense_constraints(
            query_ids,
            query_data,
            result_data,
            verbose=False,
            lang=args.lang,
        )
    )
    (
        _,
        _,
        _,
        conditional_micro_logi,
        _,
        logical_pass_ids,
    ) = evaluate_hard_constraints_v2(
        query_ids,
        query_data,
        result_data,
        env_pass_id=commonsense_pass_ids,
        verbose=False,
        lang=args.lang,
    )

    all_pass_ids = list(
        set(schema_pass_ids)
        & set(commonsense_pass_ids)
        & set(logical_pass_ids)
    )
    fpr = 100.0 * len(all_pass_ids) / len(query_ids)
    pre_res = cal_default_pr_score(
        query_ids, query_data, result_data, all_pass_ids
    )

    # Keep the official eval_tpc.py field definitions and score formula.
    scores = {
        "MicEPR": micro_comm,
        "MacEPR": macro_comm,
        "C-LPR": conditional_micro_logi,
        "FPR": fpr,
        "DAV": pre_res[0] * 100,
        "ATT": pre_res[1] * 100,
        "DDR": pre_res[2] * 100,
    }
    scores["overall"] = (
        0.1 * micro_comm
        + 0.1 * micro_comm
        + 0.25 * conditional_micro_logi
        + 0.05 * scores["DAV"]
        + 0.05 * scores["ATT"]
        + 0.05 * scores["DDR"]
        + 0.4 * fpr
    )

    report = {
        "mode": "partial",
        "method": args.method,
        "split": args.splits,
        "evaluated_count": len(query_ids),
        "split_count": len(all_query_ids),
        "coverage_percent": 100.0 * len(query_ids) / len(all_query_ids),
        "all_pass_count": len(all_pass_ids),
        "scores": scores,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    print("Saved: {}".format(output_path))

    if args.preference:
        evaluate_preference_v2(
            query_ids,
            query_data,
            result_data,
            list(set(commonsense_pass_ids) & set(logical_pass_ids)),
            lang=args.lang,
        )
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", "-s", default="TPC_IJCAI_2026_phase1")
    parser.add_argument(
        "--method", "-m", default="UrbanTrip_TPCLLM_en_oracletranslation"
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Override results/<method> with another result directory.",
    )
    parser.add_argument("--lang", "--locale", choices=["zh", "en"], default="en")
    parser.add_argument("--preference", "-p", action="store_true")
    parser.add_argument("--output", default="partial_tpc_scores.json")
    return parser.parse_args(argv)


def main():
    evaluate_partial(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
