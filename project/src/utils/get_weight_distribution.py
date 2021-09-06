import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)


def get_distribution_weights(train: np.ndarray) -> torch.FloatTensor:
    train = train.tolist()
    train_0 = [label for label in train if label == 0]
    train_1 = [label for label in train if label == 1]
    logger.debug(f"Class distribution -- train: {len(train_0)} 0's, {len(train_1)} 1's")

    n_samples = [len(train_0), len(train_1)]
    normed_weights = [1 - (x / sum(n_samples)) for x in n_samples]
    normed_weights = torch.FloatTensor(normed_weights)

    return normed_weights
