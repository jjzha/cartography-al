import argparse
import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(self,
                 args: argparse.Namespace,
                 vocab_size: int,
                 emb_dim: int,
                 num_labels: int,
                 pt_emb: np.ndarray = None) -> None:
        super().__init__()
        self.args = args
        if args.pretrained:
            pt_emb = torch.FloatTensor(pt_emb)
            if args.freeze:
                self.embedding = nn.Embedding.from_pretrained(pt_emb, freeze=True)
            else:
                self.embedding = nn.Embedding.from_pretrained(pt_emb, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, emb_dim)
        self.fc3 = nn.Linear(emb_dim, num_labels)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x) -> torch.Tensor:
        embeds = torch.sum(self.embedding(x), dim=1)
        out_fc1 = F.relu(self.fc1(embeds))

        if self.args.acquisition == "bald":
            out = F.dropout(out_fc1, p=float(os.getenv("DROPOUT")), training=True)
        else:
            out = F.dropout(out_fc1, p=float(os.getenv("DROPOUT")))

        out_fc2 = F.relu(self.fc2(out))

        if self.args.acquisition == "bald":
            out = F.dropout(out_fc2, p=float(os.getenv("DROPOUT")), training=True)
        else:
            out = F.dropout(out_fc2, p=float(os.getenv("DROPOUT")))

        out_fc3 = self.fc3(out)

        return self.log_softmax(out_fc3)

    def forward_discriminative(self, x) -> torch.Tensor:
        embeds = torch.sum(self.embedding(x), dim=1)
        out_fc1 = F.relu(self.fc1(embeds))
        out = F.dropout(out_fc1, p=float(os.getenv("DROPOUT")))
        out_fc2 = F.relu(self.fc2(out))
        out = F.dropout(out_fc2, p=float(os.getenv("DROPOUT")))

        return out

    def predict_class(self, pred: torch.Tensor) -> List:
        class_outputs = []

        for output in pred:
            output = torch.exp(output)
            class_outputs.append(output.tolist().index(max(output.tolist())))

        return class_outputs

    def predict_proba(self, pred: torch.Tensor) -> List:
        softmax_outputs = []

        for output in pred:
            softmax_outputs.append(F.softmax(output, dim=0))

        return softmax_outputs

    def weight_reset(self) -> None:
        reset_parameters = getattr(self, "reset_parameters", None)
        if callable(reset_parameters):
            self.reset_parameters()
