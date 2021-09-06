import argparse
import os

import numpy as np
import pandas as pd
from deepsig import aso


# https://github.com/Kaleidophon/deep-significance/tree/main/deepsig

def check_significance(args: argparse.Namespace):
    M = 6  # Number of different models / algorithms
    num_comparisons = M * (M - 1) / 2
    eps_min = np.eye(M)  # M x M matrix with ones on diagonal

    strategy_order = {"random"         : [],
                      "leastconfidence": [],
                      "entropy"        : [],
                      "bald"           : [],
                      "discriminative" : [],
                      "cartography"    : []}

    for entry in os.scandir(f"{os.getenv('RESULTS_PATH')}{args.task}"):
        if entry.path.endswith(".csv") and not entry.path.endswith("analysis.csv"):
            strategy = entry.path.split("/")[-1].split("_")[0]
            with open(entry.path) as f:
                df = pd.read_csv(f, sep="\t")
                scores = []
                for row in df.itertuples():
                    if row.step == 1500:
                        scores.append(row.score)
                strategy_order[strategy].extend(scores)

    scores_a = list(strategy_order.values())
    scores_b = scores_a

    for i in range(M):
        for j in range(i + 1, M):
            e_min = aso(scores_a[i], scores_b[j], confidence_level=0.05 / num_comparisons)
            eps_min[i, j] = e_min
            eps_min[j, i] = 1 - e_min

    print(eps_min)
