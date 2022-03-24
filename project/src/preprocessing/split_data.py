from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def split_data(train: np.ndarray, train_size: int) -> Tuple:
    """Splits training data into initial training set and pool"""

    if train_size != len(train):
        train, pool = train_test_split(train, shuffle=True, train_size=train_size, random_state=42)
    else:
        train = train
        pool = []

    return train, pool
