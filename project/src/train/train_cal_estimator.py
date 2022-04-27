import argparse
import logging
import os
from typing import List
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from project.src.estimators.dal_estimator import DALMLP
from project.src.utils.data_loader import DatasetMapperDiscriminative
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'


class CALEstimator:
    def __init__(self, args: argparse.Namespace, current_size: int, emb_dim: int, num_labels: int,
                 class_weights: torch.FloatTensor) -> None:
        self.model = DALMLP(emb_dim, num_labels).to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.current_size = current_size
        self.args = args
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate_binary)
        logging.info(f"Optimizing CAL Classifier using {self.optimizer.__class__.__name__} with learning rate "
                     f"{self.args.learning_rate_binary}.")

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        train = DatasetMapperDiscriminative(torch.from_numpy(X_train), torch.from_numpy(y_train))
        loader_train = DataLoader(train, batch_size=self.args.batch_size)
        self.model.train()
        
        for epoch in range(self.args.epochs):
            epoch_loss = 0
            y_pred = []
            y_gold = []

            for batch_x, batch_y, idx in loader_train:
                self.model.zero_grad()

                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                raw_logits = self.model.forward(batch_x)
                predictions = self.model.predict_class(raw_logits)

                y_pred.extend(predictions)
                y_gold.extend(batch_y.tolist())

                loss = self.criterion(raw_logits, batch_y)
                loss.backward()

                self.optimizer.step()
                epoch_loss += loss

            sys.stdout.write(f"\rCAL Classifier: Epoch {epoch}, train loss: {epoch_loss / len(y_gold):.4f}"
                             f", accuracy: {accuracy_score(y_pred, y_gold):.4f}")
            sys.stdout.flush()
        print("\n", end='')

    def predict(self, X_pool: np.ndarray, y_pool: np.ndarray) -> list:
        pool = DatasetMapperDiscriminative(torch.from_numpy(X_pool), torch.from_numpy(y_pool))
        loader_pool = DataLoader(pool, batch_size=self.args.batch_size)
        probas = []
        indices = []

        self.model.eval()
        for batch_x, _, idx in loader_pool:
            batch_x = batch_x.to(DEVICE)
            indices.extend(idx)

            with torch.no_grad():
                raw_logits = self.model.forward(batch_x)
                probas.extend(self.model.predict_proba(raw_logits))

        idx_with_probas = [(idx, proba) for idx, proba in zip(indices, probas)]
        idx_with_probas.sort(key=lambda tup: (torch.amax(tup[1])))
        top_k_indices = [idx.item() for idx, proba in idx_with_probas[:int(os.getenv("ACTIVE_LEARNING_BATCHES"))]]

        return top_k_indices

    def weight_reset(self) -> None:
        self.model.weight_reset()

    def predict_class(self, predictions: torch.Tensor) -> List:
        return self.model.predict_class(predictions)
