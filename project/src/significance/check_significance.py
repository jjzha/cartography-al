import argparse
import os

import numpy as np
import pandas as pd
from deepsig import multi_aso


# https://github.com/Kaleidophon/deep-significance/tree/main/deepsig

def check_significance(args: argparse.Namespace):
    strategies = ["bald", "cartography", "discriminative", "entropy", "leastconfidence", "random"]

    strategy_order = {"random": [],
                      "leastconfidence": [],
                      "entropy": [],
                      "bald": [],
                      "discriminative": [],
                      "cartography": []}

    for strat in strategies:
        for entry in os.scandir(f"{os.getenv('RESULTS_PATH')}{args.task}"):
            strategy = entry.path.split("/")[-1].split("_")[0]
            if entry.path.endswith(".csv") and strategy == strat:
                with open(entry.path) as f:
                    df = pd.read_csv(f, sep="\t")
                    scores = []
                    for row in df.itertuples():
                        scores.append(row.score)
                    strategy_order[strategy].extend(scores)

    e_min = multi_aso(strategy_order, confidence_level=0.05, return_df=True, num_jobs=64)

    print(e_min)
