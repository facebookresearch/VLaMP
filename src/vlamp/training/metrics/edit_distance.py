# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from typing import Dict, List, Optional, Tuple
from allennlp.nn.util import dist_reduce_sum
from allennlp.training.metrics import Metric
from vlamp.training.metrics.common import PlanningMetric
import torch
import editdistance


@Metric.register("fixed-edit-distance")
class FixedTopkEditDistance(PlanningMetric):
    def __init__(self, k: int = 1, t: Optional[int] = None, pad_index: int = 0) -> None:
        super().__init__(pad_index=pad_index)
        self.total_distance = 0.0
        self.total_count = 0.0
        self.k = k
        self.t = t

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

    def get_metric(self, reset: bool) -> Dict[str, float]:
        ed = self.total_distance / self.total_count if self.total_count > 0 else 0
        if reset:
            self.reset()
        return {"ed": ed}

    def pick_top_k(self, predictions: torch.Tensor) -> torch.Tensor:
        return predictions[:, : self.k, :]

    def call(
        self,
        predictions: torch.Tensor,  # (batch, k, seq_len)
        gold_labels: torch.Tensor,
        mask: torch.BoolTensor,
    ):

        batch, K, seq_len = predictions.shape
        total_distance = 0.0
        total_count = 0.0
        for n in range(batch):
            total_distance += min(
                [
                    editdistance.eval(
                        predictions[n, k, :].tolist(), gold_labels[n, k, :].tolist()
                    )
                    / seq_len
                    for k in range(K)
                ]
            )
            total_count += 1
        self.total_count += int(dist_reduce_sum(total_count))
        self.total_distance += float(dist_reduce_sum(total_distance))

    def reset(self) -> None:
        self.total_distance = 0.0
        self.total_count = 0.0


@Metric.register("edit-distance")
class EditDistance(Metric):
    def __init__(
        self, top_ks_and_ts: Optional[List[Tuple[int, int]]] = None, pad_index: int = 0
    ) -> None:
        super().__init__()
        if top_ks_and_ts is None:
            top_ks_and_ts = [(1, 3), (1, 4)]
        self.ed_metrics = [
            FixedTopkEditDistance(*top_k_and_t, pad_index=pad_index)
            for top_k_and_t in top_ks_and_ts
        ]

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        for metric in self.ed_metrics:
            metric(predictions, gold_labels, mask)

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        return {
            (
                f"top_{metric.k}_ed_at_t_"
                f"{metric.t if metric.t is not None else 'full'}"
            ): metric.get_metric(reset)["ed"]
            for metric in self.ed_metrics
        }

    def reset(self) -> None:
        for metric in self.ed_metrics:
            metric.reset()
