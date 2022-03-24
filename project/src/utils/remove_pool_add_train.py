import json
import logging
from collections import Counter
from typing import List, Tuple

import numpy as np


def add_and_remove_instances(X_train: np.ndarray, y_train: np.ndarray, X_pool: np.ndarray, y_pool: np.ndarray,
                             top_k_indices: List) -> Tuple:
    X_train = np.concatenate((X_train, X_pool[top_k_indices]))
    y_train = np.concatenate((y_train, y_pool[top_k_indices]))
    logging.info(f"Selected Instances:"
                 f" {json.dumps(dict(sorted(Counter(y_pool[top_k_indices].tolist()).items())), indent=4)}")

    X_pool = np.delete(X_pool, top_k_indices, 0)
    y_pool = np.delete(y_pool, top_k_indices, 0)

    return X_train, y_train, X_pool, y_pool
