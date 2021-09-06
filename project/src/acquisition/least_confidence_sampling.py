import os
from typing import List, Tuple

import torch


def least_confidence_sampling(idx_with_probas: List[Tuple]) -> List[str]:
    idx_with_probas.sort(key=lambda tup: (1 - torch.max(tup[1])), reverse=True)
    indices = [idx for idx, proba in idx_with_probas][:int(os.getenv("ACTIVE_LEARNING_BATCHES"))]

    return indices
