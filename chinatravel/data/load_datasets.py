import sys
import os
import json
import numpy as np
from datasets import load_dataset as hg_load_dataset
import ast
from collections import Counter

project_root_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

from chinatravel.environment.language import normalize_lang


TPC2026_SPLITS = {
    "TPC_IJCAI_2026_phase1": (
        "TPC_IJCAI_2026_phase1",
        "TPC_IJCAI_2026_phase1_EN",
    ),
    "TPC_IJCAI_2026_phase1_EN": (
        "TPC_IJCAI_2026_phase1",
        "TPC_IJCAI_2026_phase1_EN",
    ),
    "tpc2026_phase1": (
        "TPC_IJCAI_2026_phase1",
        "TPC_IJCAI_2026_phase1_EN",
    ),
    "tpc_phase1": (
        "TPC_IJCAI_2026_phase1",
        "TPC_IJCAI_2026_phase1_EN",
    ),
}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def _local_split_paths(args):
    split = str(args.splits)
    split_name, split_data_dir = TPC2026_SPLITS.get(split, (split, None))
    split_config_file = os.path.join(
        project_root_path,
        "chinatravel",
        "evaluation",
        "default_splits",
        f"{split_name}.txt",
    )

    explicit_query_dir = getattr(args, "query_dir", None)
    if explicit_query_dir:
        data_dir = os.path.abspath(os.path.expanduser(explicit_query_dir))
    elif split_data_dir:
        data_dir = os.path.join(
            project_root_path, "chinatravel", "data", split_data_dir
        )
    else:
        data_dir = os.path.join(project_root_path, "chinatravel", "data")
        lang = normalize_lang(getattr(args, "lang", None))
        if lang == "en":
            data_dir = os.path.join(data_dir, "en")

    return split_config_file, data_dir


def load_query_local(args, version="", verbose=False):
    query_data = {}
    split_config_file, data_dir = _local_split_paths(args)

    print("config file for testing split: {}".format(split_config_file))
    print("query data directory: {}".format(data_dir))

    if not os.path.isfile(split_config_file):
        raise FileNotFoundError(f"Split index not found: {split_config_file}")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Query data directory not found: {data_dir}")

    query_id_list = []
    with open(split_config_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                query_id_list.append(line)

    duplicate_ids = sorted(
        query_id for query_id, count in Counter(query_id_list).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(
            "Duplicate query ids in split index: {}".format(
                ", ".join(duplicate_ids[:10])
            )
        )

    if verbose:
        print(query_id_list)

    search_dirs = [data_dir]
    search_dirs.extend(
        os.path.join(data_dir, name)
        for name in sorted(os.listdir(data_dir))
        if os.path.isdir(os.path.join(data_dir, name))
    )

    missing_ids = []
    for query_id in query_id_list:
        query_path = next(
            (
                os.path.join(directory, f"{query_id}.json")
                for directory in search_dirs
                if os.path.isfile(os.path.join(directory, f"{query_id}.json"))
            ),
            None,
        )
        if query_path is None:
            missing_ids.append(query_id)
            continue

        with open(query_path, "r", encoding="utf-8") as query_file:
            data_i = json.load(query_file)
        if str(data_i.get("uid")) != query_id:
            raise ValueError(
                f"Query uid mismatch: index={query_id}, file={query_path}, "
                f"json_uid={data_i.get('uid')!r}"
            )

        if hasattr(args, "oracle_translation") and not args.oracle_translation:
            data_i.pop("hard_logic", None)
            data_i.pop("hard_logic_py", None)
            data_i.pop("hard_logic_nl", None)
        query_data[query_id] = data_i

    if missing_ids:
        raise FileNotFoundError(
            "Missing {} query files under {}. First ids: {}".format(
                len(missing_ids), data_dir, ", ".join(missing_ids[:10])
            )
        )

    # print(query_data)

    if verbose:
        for query_id in query_id_list:
            print(query_id, query_data[query_id])

    return query_id_list, query_data


def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(json_data, file_path):
    with open(file_path, "w", encoding="utf8") as dump_f:
        json.dump(json_data, dump_f, ensure_ascii=False, indent=4, cls=NpEncoder)



def load_query(args):
    split = str(args.splits)
    lang_value = getattr(args, "lang", None)
    if lang_value is None and split in TPC2026_SPLITS:
        lang_value = "en"
    lang = normalize_lang(lang_value)
    if lang == "en" or split in TPC2026_SPLITS or getattr(args, "query_dir", None):
        return load_query_local(args)

    if not args.splits in ["easy", "medium", "human", "preference_base50",
                           "preference0_base50", "preference1_base50", "preference2_base50",
                           "preference3_base50", "preference4_base50", "preference5_base50"]:
        return load_query_local(args)
    config_name = "default"
    if args.splits in ["preference0_base50", "preference1_base50", "preference2_base50",
                       "preference3_base50", "preference4_base50", "preference5_base50"]:
        config_name = "preference"
    # elif args.splits in ["human"]:
    #     config_name = "validation"
    # elif args.splits in ["human1000"]:
    #     config_name = "test"
    query_data = hg_load_dataset("LAMDA-NeSy/ChinaTravel", name=config_name)[args.splits].to_list()


    for data_i in query_data:
        if "hard_logic_py" in data_i:
            data_i["hard_logic_py"] = ast.literal_eval(data_i["hard_logic_py"])

    query_id_list = [data_i["uid"] for data_i in query_data]
    data_dict = {}
    for data_i in query_data:
        if not getattr(args, "oracle_translation", False):
            if "hard_logic" in data_i:
                del data_i["hard_logic"]
            if "hard_logic_py" in data_i:
                del data_i["hard_logic_py"]
            if "hard_logic_nl" in data_i:
                del data_i["hard_logic_nl"]

        data_dict[data_i["uid"]] = data_i

    return query_id_list, data_dict


import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("--splits", type=str, default="easy")
argparser.add_argument("--lang", type=str, choices=["zh", "en"], default="zh")

if __name__ == "__main__":


    # from datasets import load_dataset as hg_load_dataset

    # # Login using e.g. `huggingface-cli login` to access this dataset
    # ds = hg_load_dataset("LAMDA-NeSy/ChinaTravel")
    # print(ds)
    # print(ds["easy"].to_list())

    # exit(0)
    args = argparser.parse_args()
    query_id_list, query_data = load_query(args)
    # print(query_id_list)
    # print(query_data)

    for uid in query_id_list:
        if uid in query_data:
            print(uid, query_data[uid])
        else:
            raise ValueError(f"{uid} not in query_data")
