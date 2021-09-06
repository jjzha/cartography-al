import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


logger = logging.getLogger(__name__)


def check_overlap_indices(args):
    total_results = {}
    for entry in os.scandir(f"{os.getenv('INDICES_PATH')}"):
        if args.task in entry.path:
            method = entry.path.split("/")[-1].split("_")[-1].split(".")[0]
            with open(entry.path, "r") as f:
                data = json.load(f)
                for interval, indices in data.items():
                    if not total_results.get(method):
                        total_results[method] = {
                                interval: {"seed_1": indices[0], "seed_2": indices[1], "seed_3": indices[2],
                                           "seed_4": indices[3], "seed 5": indices[4]}}
                    elif interval not in total_results.get(method):
                        total_results[method].update(
                                {interval: {"seed_1": indices[0], "seed_2": indices[1], "seed_3": indices[2],
                                            "seed_4": indices[3], "seed 5": indices[4]}})
                    elif "seed_1" not in total_results[method].get(interval):
                        total_results[method][interval].update(
                                {"seed_1": indices[0], "seed_2": indices[1], "seed_3": indices[2], "seed_4": indices[3],
                                 "seed 5": indices[4]})

    df_total_results = pd.DataFrame.from_dict(total_results)
    logger.info(df_total_results)

    d = {}
    for row in df_total_results.itertuples():
        for seed, dal, cal, lc, rand in zip(row.discriminative,
                                            row.discriminative.values(),
                                            row.cartography.values(),
                                            row.leastconfidence.values(),
                                            row.random.values()):
            random = set(rand)
            cartography = set(cal)
            discriminative = set(dal)
            leastconfidence = set(lc)

            if seed not in d:
                d[seed] = {"RANDLC" : [len(random.intersection(leastconfidence))],
                           "RANDDAL": [len(random.intersection(discriminative))],
                           "RANDCAL": [len(random.intersection(cartography))],
                           "CALLC"  : [len(cartography.intersection(leastconfidence))],
                           "CALDAL" : [len(cartography.intersection(discriminative))],
                           "DALLC"  : [len(discriminative.intersection(leastconfidence))]}

            else:
                d[seed]["CALLC"].append(len(cartography.intersection(leastconfidence)))
                d[seed]["CALDAL"].append(len(cartography.intersection(discriminative)))
                d[seed]["DALLC"].append(len(discriminative.intersection(leastconfidence)))
                d[seed]["RANDLC"].append(len(random.intersection(leastconfidence)))
                d[seed]["RANDDAL"].append(len(random.intersection(discriminative)))
                d[seed]["RANDCAL"].append(len(random.intersection(cartography)))

    total = {"RANDLC": 0, "RANDDAL": 0, "RANDCAL": 0, "DALLC": 0, "CALLC": 0, "CALDAL": 0}
    for seed, dic in d.items():
        for method, values in dic.items():
            for val in values:
                total[method] += val

    logging.info(f"Overlapping instances {args.task} over {len(os.getenv('SEEDS').split())} random seeds: {total}")
    total_instances_for_al = int(os.getenv('ITERATIONS')) * int(os.getenv('ACTIVE_LEARNING_BATCHES'))
    for method, val in total.items():
        percentage_overlap = val / (total_instances_for_al * len(os.getenv('SEEDS').split()))
        logging.info(f"{method}: {'{:.5E}'.format(percentage_overlap)}")
