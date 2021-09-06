import argparse
import logging
import os
from typing import Dict, Tuple

import fasttext
import fasttext.util
import numpy as np

logger = logging.getLogger(__name__)


def get_vector_matrix(args: argparse.Namespace, word_to_idx: Dict) -> np.ndarray:
    if not os.path.isfile(f"project/resources/embeddings/embedding_matrix_{args.task}.npy"):
        if not os.path.isfile("project/resources/embeddings/cc.en.300.bin"):
            logger.info("FastText embeddings not found, downloading...")
            fasttext.util.download_model("en", if_exists="ignore")
            os.rename("cc.en.300.bin", "project/resources/embeddings/cc.en.300.bin")
            os.remove("cc.en.300.bin.gz")

        logger.info("Loading FastText embeddings...")
        ft = fasttext.load_model("project/resources/embeddings/cc.en.300.bin")

        logger.info("Getting FastText vectors from vocab")

        matrix_len = len(word_to_idx)
        weights_matrix = np.zeros((matrix_len, 300))

        for word, i in word_to_idx.items():
            weights_matrix[i] = ft[word]

        np.save(f"project/resources/embeddings/embedding_matrix_{args.task}.npy", weights_matrix)

    else:
        weights_matrix = np.load(f"project/resources/embeddings/embedding_matrix_{args.task}.npy")

    return weights_matrix


def get_vocab_and_label(train: np.ndarray, test: np.ndarray) -> Tuple[Dict, Dict, int, int]:
    word_to_idx = {}
    label_to_idx = {}
    total = np.concatenate((train, test))

    for sent, cls in total:
        for word in sent.split():
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)

        if cls not in label_to_idx:
            label_to_idx[cls] = int(cls)

    vocab_size = len(word_to_idx)
    num_labels = len(label_to_idx)

    return word_to_idx, label_to_idx, vocab_size, num_labels
