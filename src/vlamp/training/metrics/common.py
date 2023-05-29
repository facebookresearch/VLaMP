# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from typing import Optional, Tuple
from allennlp.training.metrics import Metric
import torch
import logging

logger = logging.getLogger(__name__)


def create_expanded_mask_and_gold_labels(
    predictions: torch.Tensor, gold_labels: torch.Tensor, pad_index: int = 0
) -> Tuple[torch.Tensor, torch.BoolTensor]:
    k = predictions.size()[1]
    expanded_size = list(gold_labels.size())
    expanded_size.insert(1, k)
    expanded_gold = gold_labels.unsqueeze(1).expand(expanded_size)
    # mask out everything after max_len for each row.
    expanded_mask = ~((expanded_gold == pad_index) * (predictions == pad_index))
    return expanded_gold, expanded_mask


class PlanningMetric(Metric):
    def __init__(self, pad_index: int = 0) -> None:
        super().__init__()
        self.pad_index = pad_index

    def reset(self):
        raise NotImplementedError

    def _check_shape(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
    ) -> None:
        assert (
            len(predictions.shape) == 3
        ), f"If k={self.k}, then predictions should have 3 dimensions with shape[1]>={self.k}"
        assert (
            predictions.shape[1] >= self.k
        ), f"predictions.shape[1] should be >= {self.k} but is {predictions.shape[1]}"
        assert predictions.shape[-1] == gold_labels.shape[-1]

    def pick_top_k(self, predictions: torch.Tensor) -> torch.Tensor:
        return predictions

    def adjust_length(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        raise NotImplementedError

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> None:
        predictions, gold_labels, mask = self.detach_tensors(
            predictions, gold_labels, mask
        )
        self._check_shape(predictions, gold_labels)
        if mask is not None:
            logger.warning("mask will be ignored")
        predictions = self.pick_top_k(predictions)
        # predictions = predictions[:, : self.k, :]
        gold_labels, mask = create_expanded_mask_and_gold_labels(
            predictions, gold_labels, self.pad_index
        )
        assert predictions.shape == gold_labels.shape
        predictions, gold_labels, mask = self.adjust_length(
            predictions, gold_labels, mask
        )
        self.call(predictions, gold_labels, mask)

    def call(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> None:
        raise NotImplementedError
