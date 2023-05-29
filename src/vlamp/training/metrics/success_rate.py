# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from asyncio.log import logger
from logging import warning
from typing import Dict, List, Optional, Tuple
from allennlp.common.util import pad_sequence_to_length
from allennlp.nn.util import dist_reduce_sum
from allennlp.training.metrics import Metric, SequenceAccuracy
import torch
from .common import PlanningMetric, create_expanded_mask_and_gold_labels


@Metric.register("fixed-topk-sequence-accuracy")
class FixedTopkSequenceAccuracy(PlanningMetric):
    def __init__(self, k: int = 1, t: Optional[int] = None, pad_index: int = 0) -> None:
        super().__init__(pad_index=pad_index)
        self.k = k
        self.t = t
        self.correct_count = 0.0
        self.total_count = 0.0

    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0

    def adjust_length(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        assert predictions.shape == gold_labels.shape
        if self.t:
            max_length = min(gold_labels.shape[-1], self.t)
        else:
            max_length = gold_labels.shape[-1]

        predictions = predictions[:, :, :max_length]
        gold_labels = gold_labels[:, :, :max_length]
        mask = mask[:, :, :max_length]
        return predictions, gold_labels, mask

    def pick_top_k(self, predictions: torch.Tensor) -> torch.Tensor:
        return predictions[:, : self.k, :]

    def call(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> None:

        masked_gold = gold_labels * mask
        masked_predictions = predictions * mask

        eqs = masked_gold.eq(masked_predictions)  # (batch, k, seq)
        matches_per_query = eqs.min(dim=-1)[0]  # (batch, k)
        some_match = matches_per_query.max(dim=1)[0]  # (batch,)
        correct = some_match.sum().item()
        _total_count = predictions.size()[0]
        _correct_count = correct

        self.correct_count += dist_reduce_sum(_correct_count)
        self.total_count += dist_reduce_sum(_total_count)

    def get_metric(self, reset: bool = False):
        """
        # Returns
        The accumulated accuracy.
        """
        if self.total_count > 0:
            accuracy = self.correct_count / self.total_count
        else:
            accuracy = 0
        if reset:
            self.reset()
        return {"accuracy": accuracy}


@Metric.register("success-rate")
class SuccessRate(Metric):
    def __init__(
        self, top_ks_and_ts: Optional[List[Tuple[int, int]]] = None, pad_index: int = 0
    ) -> None:
        super().__init__()
        if top_ks_and_ts is None:
            top_ks_and_ts = [(1, 3), (1, 4)]
        self.seq_accuracy_metrics = [
            FixedTopkSequenceAccuracy(*top_k_and_t, pad_index=pad_index)
            for top_k_and_t in top_ks_and_ts
        ]

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        for metric in self.seq_accuracy_metrics:
            metric(predictions, gold_labels, mask)

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        return {
            (
                f"top_{metric.k}_sr_at_t_"
                f"{metric.t if metric.t is not None else 'full'}"
            ): metric.get_metric(reset)["accuracy"]
            for metric in self.seq_accuracy_metrics
        }

    def reset(self) -> None:
        for metric in self.seq_accuracy_metrics:
            metric.reset()
