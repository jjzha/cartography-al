import logging

import numpy as np


def get_statistics_datamap(confidence_run: list, variability_run: list, correctness_run: list) -> None:
    confidence = np.mean(confidence_run, axis=0)
    variability = np.mean(variability_run, axis=0)
    correctness = np.mean(correctness_run, axis=0)
    # for latex
    logging.info(f"confidence: {' & '.join([str(round(conf, 3)) for conf in confidence])} & {np.mean(confidence)}")
    logging.info(f"variability: {' & '.join([str(round(var, 3)) for var in variability])} & {np.mean(variability)}")
    logging.info(f"correctness: {' & '.join([str(round(cor, 3)) for cor in correctness])} & {np.mean(correctness)}")
