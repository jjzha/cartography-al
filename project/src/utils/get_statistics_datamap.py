import logging

import numpy as np

logger = logging.getLogger(__name__)


def get_statistics_datamap(confidence_run: list, variability_run: list, correctness_run: list) -> None:
    confidence = np.mean(confidence_run, axis=0)
    variability = np.mean(variability_run, axis=0)
    correctness = np.mean(correctness_run, axis=0)
    # for latex
    print(f"confidence: {' & '.join([str(round(conf, 3)) for conf in confidence])} & {np.mean(confidence)}")
    print(f"variability: {' & '.join([str(round(var, 3)) for var in variability])} & {np.mean(variability)}")
    print(f"correctness: {' & '.join([str(round(cor, 3)) for cor in correctness])} & {np.mean(correctness)}")
