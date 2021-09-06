from typing import Tuple
import os
import numpy as np
from project.src.train.generate_cartography import transform_correctness_to_bins


def prepare_data_for_cal(X_train_rep: list, X_pool_rep: list, correctness: dict) -> Tuple:
    correctness = {idx: transform_correctness_to_bins(correct) for idx, correct in list(correctness.items())}
    X_train, y_train = [], []
    for idx, label in correctness.items():
        X_train.append(X_train_rep[int(idx)])

        if label <= float(os.getenv("CAL_THRESHOLD")):
            y_train.append(0)
        else:
            y_train.append(1)

    X_pool, y_pool = np.array(X_pool_rep), np.zeros(len(X_pool_rep))
    X_train, y_train = np.array(X_train), np.array(y_train)

    return X_train, y_train, X_pool, y_pool
