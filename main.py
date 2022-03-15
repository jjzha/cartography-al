import argparse
import json
import logging
import os
import random
import sys
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

load_dotenv(verbose=True)


def setup_experiment(args, out_path):
    if not os.path.exists(out_path):
        # if output dir does not exist, create it (new experiment)
        print(f"Path '{out_path}' does not exist. Creating...")
        os.mkdir(out_path)
    # if output dir exist, check if predicting
    else:
        print(f"Path '{out_path}' already exists. Exiting...")
        exit(1)

    # setup logging
    log_format = '%(message)s'
    log_level = logging.INFO
    logging.basicConfig(filename=os.path.join(out_path, f'classify_{args.seed}.log'), filemode='a', format=log_format,
                        level=log_level)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Active Learning Experiments")
    # Task
    parser.add_argument("--task", required=True, nargs="?", choices=["trec", "agnews"])
    parser.add_argument("--analysis", action="store_true", help="Plot analysis")
    parser.add_argument("--plot_results", action="store_true", help="Plots saved data.")
    parser.add_argument("--plot_dist_results", action="store_true", help="Plots saved data.")
    parser.add_argument("--check_indices", action="store_true", help="Checks selected indices.")
    parser.add_argument("--exp_path", nargs="?", type=str, help="Path to experiment logs.")

    # Vectorization of data
    parser.add_argument("--vector", nargs="?", choices=["fasttext", "bow"])
    parser.add_argument("--pretrained", action="store_true", help="Uses pretrained embeddings.")
    parser.add_argument("--freeze", action="store_true", help="freezes pretrained embeddings.")
    # Hyperparameters
    parser.add_argument("--initial_size", nargs="?", type=int, help="Train size")
    parser.add_argument("--batch_size", nargs="?", type=int, help="batch_size")
    parser.add_argument("--al_iterations", nargs="?", type=int, default=30, help="Active Learning Iterations.")
    parser.add_argument("--seed", nargs="?", type=int, help="Seed to use.")
    parser.add_argument("--epochs", nargs="?", type=int, default=50, help="Number of epochs to make the model run.")
    parser.add_argument("--learning_rate_main", nargs="?", type=float, default=0.001, help="Learning rate for main "
                                                                                           "classifier")
    parser.add_argument("--learning_rate_binary", nargs="?", type=float, default=0.001, help="Learning rate for binary "
                                                                                             "classifier")
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
        # setup experiment directory and logging
        setup_experiment(args, args.exp_path)
        results, selected_top_k_runs = [], []
        confidence_run, variability_run, correctness_run = [], [], []

        set_seed(args.seed)
        logging.info(f"Current args.seed: {args.seed} -- Active Learning process "
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
                           "step": [step for _ in range(len(results)) for step in range(0, total, steps)]})
        logging.info(df)
        df.to_csv(
            f"{os.getenv('RESULTS_PATH')}/{args.task}/{args.acquisition}_{args.initial_size}_{args.seed}.csv",
            sep="\t")
