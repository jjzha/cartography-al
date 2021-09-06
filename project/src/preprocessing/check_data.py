import argparse
import logging
import os
from typing import Tuple

import numpy as np
from project.src.preprocessing.change_list_to_dict import change_list_to_json
from project.src.preprocessing.read_data import read_data

logger = logging.getLogger(__name__)


def check_and_get_data(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    if args.task == "trec":
        train_path, test_path = os.getenv("TREC_TRAIN"), os.getenv("TREC_TEST")
    elif args.task == "agnews":
        train_path, test_path = os.getenv("AGNEWS_TRAIN"), os.getenv("AGNEWS_TEST")
    else:
        train_path, test_path = None, None
        logging.warning("Reading in data went wrong!")

    train, test = read_data(train_path, test_path)
    train, test = change_list_to_json(train, test)

    return train, test
