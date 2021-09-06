import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DALMLP(nn.Module):
    def __init__(self, emb_dim: int, num_labels: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, num_labels)

    def forward(self, x) -> torch.Tensor:
        out_fc1 = F.relu(self.fc1(x))

        return out_fc1

    def predict_class(self, pred: torch.Tensor) -> List:
        class_outputs = []

        for output in pred:
            output = F.softmax(output, dim=0)
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
