import argparse
import os

import numpy as np
import pandas as pd
from deepsig import multi_aso


# https://github.com/Kaleidophon/deep-significance/tree/main/deepsig

def check_significance(args: argparse.Namespace):
    M = 6  # Number of different models / algorithms
    num_comparisons = M * (M - 1) / 2
    eps_min = np.eye(M)  # M x M matrix with ones on diagonal

    strategy_order = {"random": [],
                      "leastconfidence": [],
                      "entropy": [],
                      "bald": [],
                      "discriminative": [],
                      "cartography": []}

    for entry in os.scandir(f"{os.getenv('RESULTS_PATH')}{args.task}"):
        if entry.path.endswith(".csv") and not entry.path.endswith("analysis.csv"):
            strategy = entry.path.split("/")[-1].split("_")[0]
            with open(entry.path) as f:
                df = pd.read_csv(f, sep="\t")
                scores = []
                for row in df.itertuples():
                    if row.step == 1450:
                        scores.append(row.score)
                strategy_order[strategy].extend(scores)

    e_min = multi_aso(strategy_order, confidence_level=0.05, return_df=True, num_jobs=64)

    print(e_min)
