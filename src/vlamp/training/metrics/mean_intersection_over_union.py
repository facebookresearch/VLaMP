# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from typing import Dict, List, Optional, Tuple
from allennlp.nn.util import dist_reduce_sum
from allennlp.training.metrics import Metric
from vlamp.training.metrics.common import PlanningMetric
import torch
import torch.nn.functional as F


@Metric.register("fixed-topk-miou")
class FixedTopkMeanIntersectionOverUnion(PlanningMetric):
    def __init__(self, k: int = 1, t: Optional[int] = None, pad_index: int = 0) -> None:
        super().__init__(pad_index=pad_index)
        self.iou = 0.0
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
        miou = self.iou / self.total_count if self.total_count > 0 else 0
        if reset:
            self.reset()
        return {"miou": miou}

    def pick_top_k(self, predictions: torch.Tensor) -> torch.Tensor:
        return predictions[:, : self.k, :]

    def call(
        self,
        predictions: torch.Tensor,  # (batch, k, seq_len)
        gold_labels: torch.Tensor,
        mask: torch.BoolTensor,
    ):

        largest_id = max(int(predictions.max()), int(gold_labels.max())) + 1
        one_hot_predictions = F.one_hot(
            predictions, largest_id
        )  # (batch, k, seq_len, num_classes)
        one_hot_labels = F.one_hot(
            gold_labels, largest_id
        )  # (batch, k, seq_len, num_classes)
        mask_expanded = (
            mask.unsqueeze(-1).expand_as(one_hot_labels).contiguous()
        )  # (batch, k, seq_len, 1)
        mask_expanded[:, :, :, self.pad_index] = 0
        intersection_count = (
            (
                ((one_hot_predictions * mask_expanded).sum(dim=-2) > 0)
                * ((one_hot_labels * mask_expanded).sum(dim=-2) > 0)
            )
        ).sum(
            -1
        )  # (batch, k)
        union_count = (
            (
                ((one_hot_predictions * mask_expanded).sum(dim=-2) > 0)
                + ((one_hot_labels * mask_expanded).sum(dim=-2) > 0)
            )
        ).sum(
            -1
        )  # (batch, k)
        sum_iou = ((intersection_count / union_count).max(dim=-1)[0]).sum()
        total_count = gold_labels.shape[0]
        self.total_count += int(dist_reduce_sum(total_count))
        self.iou += float(dist_reduce_sum(sum_iou))

    def reset(self) -> None:
        self.iou = 0.0
        self.total_count = 0.0


@Metric.register("miou")
class MeanIntersectionOverUnion(Metric):
    def __init__(
        self, top_ks_and_ts: Optional[List[Tuple[int, int]]] = None, pad_index: int = 0
    ) -> None:
        super().__init__()
        if top_ks_and_ts is None:
            top_ks_and_ts = [(1, 3), (1, 4)]
        self.miou_metrics = [
            FixedTopkMeanIntersectionOverUnion(*top_k_and_t, pad_index=pad_index)
            for top_k_and_t in top_ks_and_ts
        ]

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        for metric in self.miou_metrics:
            metric(predictions, gold_labels, mask)

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        return {
            (
                f"top_{metric.k}_miou_at_t_"
                f"{metric.t if metric.t is not None else 'full'}"
            ): metric.get_metric(reset)["miou"]
            for metric in self.miou_metrics
        }

    def reset(self) -> None:
        for metric in self.miou_metrics:
            metric.reset()
