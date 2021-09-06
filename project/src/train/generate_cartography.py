import argparse
import logging
from typing import Dict, List

import numpy as np
from project.src.utils.save_cartography import plot_cartography, save_cartography

logger = logging.getLogger(__name__)


def transform_correctness_to_bins(numbers: List) -> float:
    """to interval of 0.2"""
    fraction = sum(numbers) / len(numbers)

    return round(fraction * 5) / 5


def generate_cartography(cartography: Dict, probabilities: Dict, correctness: Dict) -> Dict:
    confidences = {idx: sum(proba) / len(proba) for idx, proba in list(probabilities.items())}
    variability = {idx: np.std(proba) for idx, proba in list(probabilities.items())}
    correctness = {idx: transform_correctness_to_bins(correct) for idx, correct in list(correctness.items())}

    logger.info(f"Length of confidences, variability, correctness: "
                f"{len(confidences)}, {len(variability)}, {len(correctness)}")

    for idx, value in confidences.items():
        cartography["confidence"].append(value)
        cartography["variability"].append(variability[idx])
        cartography["correctness"].append(correctness[idx])

    return cartography


def generate_cartography_by_idx(cartography: Dict, probabilities: Dict, correctness: Dict) -> Dict:
    confidences = {idx: sum(proba) / len(proba) for idx, proba in list(probabilities.items())}
    variability = {idx: np.std(proba) for idx, proba in list(probabilities.items())}
    correctness = {idx: transform_correctness_to_bins(correct) for idx, correct in list(correctness.items())}

    logger.info(f"Length of confidences, variability, correctness: "
                f"{len(confidences)}, {len(variability)}, {len(correctness)}")

    for idx, value in confidences.items():
        cartography[idx] = [value, variability[idx], correctness[idx]]

    return cartography


def generate_cartography_after_intervals(args, cartography: Dict) -> None:
    logger.info(f"Plotting cartography...")
    plot_cartography(args, cartography)
