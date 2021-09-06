import logging
from typing import Any, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def read_data(train_path: str, test_path: str) -> Tuple[Any, Any]:
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    train_data["Description"] = train_data["Description"].map(lambda x: x.replace("\\", ""))
    test_data["Description"] = test_data["Description"].map(lambda x: x.replace("\\", ""))

    train = train_data[["Description", "Class Index"]].values.tolist()
    test = test_data[["Description", "Class Index"]].values.tolist()

    logging.info(f"Number of instances -- train: {len(train)}, test: {len(test)}")

    return train, test
