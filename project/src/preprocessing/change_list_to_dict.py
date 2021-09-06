import logging
from typing import Any, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def change_list_to_json(train: List, test: List) -> Tuple[Any, Any]:
    train_array = np.empty(len(train), dtype=object)
    test_array = np.empty(len(test), dtype=object)

    train = [(sent, cls - 1) for (sent, cls) in train]
    test = [(sent, cls - 1) for (sent, cls) in test]

    train_array[:] = train
    test_array[:] = test

    return train_array, test_array
