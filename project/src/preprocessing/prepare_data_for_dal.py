from typing import Tuple

import numpy as np


def prepare_data_for_dal(X_train_rep: list, X_pool_rep: list) -> Tuple:
    y_train_labeled = [1 for _ in range(len(X_train_rep))]
    y_train_unlabeled = [0 for _ in range(len(X_pool_rep))]

    X_train = X_train_rep + X_pool_rep
    y_train = y_train_labeled + y_train_unlabeled

    X_train, y_train = np.array(X_train), np.array(y_train)

    return X_train, y_train
