# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from typing import Dict, List, Optional, Tuple
from allennlp.nn.util import dist_reduce_sum
from allennlp.training.metrics import Metric
from numpy import pad
from vlamp.training.metrics.common import PlanningMetric
import torch


class MicroAccuracy(PlanningMetric):
    def __init__(self, t: Optional[int] = None, pad_index: int = 0) -> None:
        super().__init__(pad_index=pad_index)
        self.k = 1
        self.t = t
        self.correct_count = 0.0
        self.total_count = 0.0

    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0

    def get_metric(self, reset: bool):
        acc = self.correct_count / self.total_count if self.total_count else 0
        if reset:
            self.reset()
        return {"accuracy": acc}

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
        _correct_count = (predictions.eq(gold_labels) * mask).unsqueeze(1).sum().item()
        _total_count = mask.sum().item()

        self.correct_count += dist_reduce_sum(_correct_count)
        self.total_count += dist_reduce_sum(_total_count)


@Metric.register("planning-accuracy")
class PlanningAccuracy(Metric):
    def __init__(self, ts: Optional[List[int]] = None, pad_index: int = 0) -> None:
        super().__init__()
        if ts is None:
            ts = [3, 4]
        self.acc_metrics = [MicroAccuracy(t, pad_index=pad_index) for t in ts]

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        for metric in self.acc_metrics:
            metric(predictions, gold_labels, mask)

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        return {
            (
                f"accuracy_at_t_" f"{metric.t if metric.t is not None else 'full'}"
            ): metric.get_metric(reset)["accuracy"]
            for metric in self.acc_metrics
        }

    def reset(self) -> None:
        for metric in self.acc_metrics:
            metric.reset()
