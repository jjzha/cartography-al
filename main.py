import argparse
import json
import logging
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn
from dotenv import load_dotenv
from project.src.active_learning import start_active_learning
from project.src.plotting.plot_results_from_csv import plot_from_csv
from project.src.significance.check_significance import check_significance
from project.src.utils.check_overlapping_indices import check_overlap_indices
from project.src.utils.get_statistics_datamap import get_statistics_datamap
from project.src.utils.save_indices_for_overlap import save_indices_for_overlap

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
load_dotenv(verbose=True)


def set_seed(seed: str) -> None:
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(int(seed))
    random.seed(int(seed))
    os.environ["PYTHONHASHSEED"] = seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Active Learning Experiments")
    # Task
    parser.add_argument("--task", required=True, nargs="?", choices=["trec", "agnews"])
    parser.add_argument("--analysis", action="store_true", help="Plot analysis")
    parser.add_argument("--plot_results", action="store_true", help="Plots saved data.")
    parser.add_argument("--plot_dist_results", action="store_true", help="Plots saved data.")
    parser.add_argument("--check_indices", action="store_true", help="Checks selected indices.")
    # Vectorization of data
    parser.add_argument("--vector", nargs="?", choices=["fasttext", "bow"])
    parser.add_argument("--pretrained", action="store_true", help="Uses pretrained embeddings.")
    parser.add_argument("--freeze", action="store_true", help="freezes pretrained embeddings.")
    # Initial trainset size
    parser.add_argument("--initial_size", nargs="?", type=int, help="Train size")
    parser.add_argument("--batch_size", nargs="?", type=int, help="batch_size")
    # Cartography specific arguments
    parser.add_argument("--cartography", action="store_true", help="Processes cartography.")
    parser.add_argument("--plot", action="store_true", help="Plots cartography.")
    parser.add_argument("--histogram", action="store_true", help="Adds density histogram to cartography.")
    parser.add_argument("--significance", action="store_true", help="Checks significance of results.")
    # Acquisition functions
    parser.add_argument("--acquisition", nargs="?",
                        choices=["random", "entropy", "leastconfidence", "bald", "cartography", "discriminative"],
                        default="random")

    args = parser.parse_args()

    if args.plot_results:
        plot_from_csv(args)
    elif args.check_indices:
        check_overlap_indices(args)
    elif args.significance:
        check_significance(args)
    else:
        results, selected_top_k_runs = [], []
        confidence_run, variability_run, correctness_run = [], [], []

        for seed in os.getenv("SEEDS").split():
            set_seed(seed)
            logging.info(f"Current seed: {seed} -- Active Learning process "
                         f"with acquisition function: [{args.acquisition}]")
            accuracy_history, selected_top_k, conf_stats, var_stats, corr_stats = start_active_learning(args)

            results.append(accuracy_history)
            selected_top_k_runs.append(selected_top_k)
            confidence_run.append(conf_stats)
            variability_run.append(var_stats)
            correctness_run.append(corr_stats)

        steps = int(os.getenv("ACTIVE_LEARNING_BATCHES"))
        total = steps * len(results[0])
        logging.info("{:30} {:30} {:30}".format("-" * 25, f"Mean accuracy of {args.acquisition}", "-" * 25))

        if args.analysis:
            print(f"Analysis of {args.acquisition}")
            save_indices_for_overlap(args, total, steps, selected_top_k_runs)
            get_statistics_datamap(confidence_run, variability_run, correctness_run)

        df = pd.DataFrame({"score": [acc for val in results for acc in val],
                           "step" : [step for _ in range(len(results)) for step in range(0, total, steps)]})
        logging.info(df)
        df.to_csv(f"{os.getenv('RESULTS_PATH')}/{args.task}/{args.acquisition}_{args.initial_size}_{os.getenv('CAL_THRESHOLD')}_caldal.csv", sep="\t")
