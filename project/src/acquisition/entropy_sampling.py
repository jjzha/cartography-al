import math
import os
from typing import List, Tuple

import torch


def entropy_sampling(idx_with_probas: List[Tuple]) -> List[str]:
    idx_with_probas.sort(key=lambda tup: (-torch.sum(tup[1] * torch.log2(tup[1]))), reverse=True)
    indices = [idx for idx, proba in idx_with_probas][:int(os.getenv("ACTIVE_LEARNING_BATCHES"))]

    return indices


def entropy_sampling_bald(idx_with_probas: List[Tuple]) -> List[str]:
    idx_with_probas.sort(key=lambda tup: -torch.sum(tup[1] * torch.log2(tup[1])), reverse=True)
    indices = [idx for idx, proba in idx_with_probas][:int(os.getenv("ACTIVE_LEARNING_BATCHES"))]

    return indices
