import argparse
import logging
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from project.src.preprocessing.check_data import check_and_get_data
from project.src.preprocessing.prepare_data_for_cal import prepare_data_for_cal
from project.src.preprocessing.prepare_data_for_dal import prepare_data_for_dal
from project.src.preprocessing.split_data import split_data
from project.src.train.generate_cartography import (generate_cartography, generate_cartography_after_intervals,
                                                    generate_cartography_by_idx, transform_correctness_to_bins)
from project.src.train.initialize_train_pool_test import initialize_train_pool_test
from project.src.train.train_cal_estimator import CALEstimator
from project.src.train.train_dal_estimator import DALEstimator
from project.src.train.train_estimator import MLPEstimator
from project.src.utils.apply_acquisition_function import apply_acquisition_function
from project.src.utils.get_vocab_and_labels import get_vector_matrix, get_vocab_and_label
from project.src.utils.get_weight_distribution import get_distribution_weights
from project.src.utils.remove_pool_add_train import add_and_remove_instances
from project.src.utils.save_cartography import save_cartography

logger = logging.getLogger(__name__)


def reset_estimator(estimator) -> None:
    estimator.weight_reset()
    estimator.probabilities.clear()
    estimator.correctness.clear()
    estimator.gold_labels.clear()


def start_active_learning(args: argparse.Namespace) -> tuple:
    logger.info("{:30} {:25} {:30}".format("-" * 25, "Initializing Data", "-" * 25))

    train, test = check_and_get_data(args)
    word_to_idx, label_to_idx, vocab_size, num_labels = get_vocab_and_label(train, test)
    train, pool = split_data(train, args.initial_size)
    embedding_matrix = get_vector_matrix(args, word_to_idx)
    emb_dim = embedding_matrix.shape[1]

    logging.info(f"Vocabulary size: {vocab_size}, "
                 f"number of labels: {num_labels}")
    if args.pretrained:
        logging.info(f"pretrained embedding size: {embedding_matrix.shape}")

    X_train, y_train, X_pool, y_pool, X_test, y_test = initialize_train_pool_test(args, train, pool, test, word_to_idx,
                                                                                  label_to_idx)

    estimator = MLPEstimator(args, vocab_size, emb_dim, num_labels, embedding_matrix)
    cartography = {"interval": [], "correctness": [], "variability": [], "confidence": []}

    # train model on initial set / create cartography
    if args.cartography:
        estimator.train(X_train, y_train)
        logger.info("{:30} {:25} {:30}".format("-" * 25, "Generating Cartography", "-" * 25))
        if args.plot:
            cartography = generate_cartography(cartography, estimator.probabilities, estimator.correctness)
            generate_cartography_after_intervals(args, cartography)
        else:
            cartography = generate_cartography_by_idx(cartography, estimator.probabilities, estimator.correctness)
            save_cartography(args, cartography)
        sys.exit(-1)
    else:
        logger.info("{:30} {:25} {:30}".format("-" * 25, "Training Model", "-" * 25))
        X_train_rep = estimator.train(X_train, y_train)
        initial_accuracy = estimator.evaluate(X_test, y_test)
        logger.info(f"Initial accuracy of estimator: {initial_accuracy}")
        logger.info("{:30} {:25} {:30}".format("-" * 25, "Starting Iterations", "-" * 25))

    active_learning_accuracy_history = [initial_accuracy]
    selected_top_k, confidence_stats, variability_stats, correctness_stats = [], [], [], []

    for i in range(int(os.getenv("ITERATIONS"))):
        logging.info(f"Active learning iteration: {i + 1}, train size: {len(X_train)}, pool size: {len(X_pool)}")

        if args.acquisition == "discriminative" or args.acquisition == "cartography":
            # prepare representations and data for DAL
            if i != 0:
                X_train_rep = estimator.train(X_train, y_train)
            X_pool_rep = estimator.predict(X_pool, y_pool)

            if args.acquisition == "discriminative":
                X_train_dal, y_train_dal = prepare_data_for_dal(X_train_rep, X_pool_rep)
                class_weights = get_distribution_weights(y_train_dal)
                dal_estimator = DALEstimator(args, len(X_train_rep), X_train_rep[0].size, len(np.unique(y_train_dal)),
                                             class_weights)
                top_k_indices = dal_estimator.train(X_train_dal, y_train_dal)
                dal_estimator.weight_reset()

            elif args.acquisition == "cartography":
                X_train_cal, y_train_cal, X_pool_cal, y_pool_cal = prepare_data_for_cal(X_train_rep, X_pool_rep,
                                                                                        estimator.correctness)
                class_weights = get_distribution_weights(y_train_cal)
                cal_estimator = CALEstimator(args, len(X_train_cal), X_train_rep[0].size, len(np.unique(y_train_cal)),
                                             class_weights)
                cal_estimator.train(X_train_cal, y_train_cal)
                top_k_indices = cal_estimator.predict(X_pool_cal, y_pool_cal)
                cal_estimator.weight_reset()

        else:
            # apply model to the pool to retrieve top-k instances
            probas = estimator.predict(X_pool, y_pool)
            top_k_indices = apply_acquisition_function(args, probas)

        # add top-k instances from pool to train and remove from pool
        X_train, y_train, X_pool, y_pool = add_and_remove_instances(X_train, y_train, X_pool, y_pool, top_k_indices)

        # retrain model, save accuracy, weight reset for next iter
        estimator.train(X_train, y_train)
        accuracy = estimator.evaluate(X_test, y_test)
        active_learning_accuracy_history.append(accuracy)

        if args.analysis:
            selected_top_k.append(top_k_indices)

            if i != 0:
                confidences = {idx: sum(proba) / len(proba) for idx, proba in list(estimator.probabilities.items())}
                variability = {idx: np.std(proba) for idx, proba in list(estimator.probabilities.items())}
                correctness = {idx: transform_correctness_to_bins(correct) for idx, correct in
                               list(estimator.correctness.items())}
                confidence_stats.append(
                    np.mean(list(confidences.values())[-int(os.getenv("ACTIVE_LEARNING_BATCHES")):]))
                variability_stats.append(
                    np.mean(list(variability.values())[-int(os.getenv("ACTIVE_LEARNING_BATCHES")):]))
                correctness_stats.append(
                    np.mean(list(correctness.values())[-int(os.getenv("ACTIVE_LEARNING_BATCHES")):]))

        reset_estimator(estimator)
        logger.info(f"Accuracy history: {active_learning_accuracy_history}")

    return active_learning_accuracy_history, selected_top_k, confidence_stats, variability_stats, correctness_stats
