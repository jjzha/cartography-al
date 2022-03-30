import argparse
import json
import logging
import os
from collections import defaultdict


def save_indices_for_overlap(args: argparse.Namespace, total: int, steps: int, selected_top_k_runs: list) -> None:
    selected_top_k_dict = defaultdict(list)

    for top_k_run in selected_top_k_runs:
        for step, idx_list in zip(range(0, total, steps), top_k_run):
            selected_top_k_dict[str(step)].append([(int(args.initial_size) + step) + idx for idx in idx_list])

    logging.info(f"Overlapping Indices: {json.dumps(selected_top_k_dict)}")
    with open(f"{os.getenv('INDICES_PATH')}{args.task}/top_k_{args.task}_{args.acquisition}_{args.seed}.json",
              "w") as f:
        json.dump(selected_top_k_dict, f)
