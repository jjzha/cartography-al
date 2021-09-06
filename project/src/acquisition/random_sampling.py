import os
from random import shuffle
from typing import List, Tuple


def random_sampling(idx_with_probas: List[Tuple]) -> List[str]:
    shuffle(idx_with_probas)
    indices = [idx for idx, proba in idx_with_probas][:int(os.getenv("ACTIVE_LEARNING_BATCHES"))]

    return indices
