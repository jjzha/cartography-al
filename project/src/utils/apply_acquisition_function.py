import argparse
from typing import List

from project.src.acquisition.entropy_sampling import entropy_sampling, entropy_sampling_bald
from project.src.acquisition.least_confidence_sampling import least_confidence_sampling
from project.src.acquisition.random_sampling import random_sampling


def apply_acquisition_function(args: argparse.Namespace, probas: list) -> List[str]:
    idx_with_probas = [(idx, proba) for idx, proba in enumerate(probas)]

    if args.acquisition == "random":
        top_k_indices = random_sampling(idx_with_probas)
    elif args.acquisition == "leastconfidence":
        top_k_indices = least_confidence_sampling(idx_with_probas)
    elif args.acquisition == "entropy":
        top_k_indices = entropy_sampling(idx_with_probas)
    elif args.acquisition == "bald":
        top_k_indices = entropy_sampling_bald(idx_with_probas)
    else:
        top_k_indices = []

    return top_k_indices
