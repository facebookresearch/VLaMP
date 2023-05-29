# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from typing import Any, Dict, List, Optional, Tuple
from allennlp.common import Registrable
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.util import (
    get_lengths_from_binary_sequence_mask,
    get_mask_from_sequence_lengths,
    get_text_field_mask,
    get_token_ids_from_text_field_tensors,
    sequence_cross_entropy_with_logits,
)
from allennlp.training.metrics import Average, CategoricalAccuracy
import numpy as np
from vlamp.training.metrics import (
    SuccessRate,
    MeanIntersectionOverUnion,
    PlanningAccuracy,
    EditDistance,
)
import torch
from torch.nn import CrossEntropyLoss


class ObservationLoss(torch.nn.Module, Registrable):
    """Base class of observation loss that works directly on predicted observation
    representations and reference representations"""

    def forward(
        self,
        obs_reps: torch.Tensor,
        target_reps: torch.Tensor,  # (bs, seq*(na+no), hidden_size)
        mask: torch.Tensor,  # (bs, seq*(na+no))
    ) -> torch.Tensor:
        """Defines the basic interface.

        Args:
            obs_reps (torch.Tensor): FloatTensor of shape (bs, seq*(na+no), hidden_size)

        Raises:
            NotImplementedError: Child class should implement the forward method.

        Returns:
            torch.Tensor: observation loss
        """
        raise NotImplementedError


@ObservationLoss.register("contrastive")
class SimpleContrastiveLoss(ObservationLoss):
    """Implements simple InfoNCE style contrastive loss."""

    def __init__(self) -> None:
        super().__init__()
        self.ce = CrossEntropyLoss()

    def forward(
        self,
        obs_reps: torch.Tensor,  # (bs, seq*(na+no), hidden_size)
        target_reps: torch.Tensor,  # (bs, seq*(na+no), hidden_size)
        mask: torch.Tensor,  # (bs, seq*(na+no))
    ) -> torch.Tensor:
        bs, num_toks, emb_size = obs_reps.shape
        mask = mask.flatten()
        obs_reps = obs_reps.view(-1, emb_size)[mask]
        target_reps = target_reps.view(-1, emb_size)[mask]
        scores = (
            obs_reps @ target_reps.T
        )  # (num_obs_tokens, num_obs_tokens) with s[i][j] = pred[i].target[j]
        loss = self.ce(
            scores,
            torch.arange(0, scores.shape[-1], dtype=torch.long, device=scores.device),
        )

        return loss


@ObservationLoss.register("mse")
class RepresentationMSELoss(ObservationLoss):
    def __init__(self) -> None:
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(
        self,
        obs_reps: torch.Tensor,  # (bs, seq*(na+no), hidden_size)
        target_reps: torch.Tensor,  # (bs, seq*(na+no), hidden_size)
        mask: torch.Tensor,  # (bs, seq*(na+no))
    ) -> torch.Tensor:
        bs, num_toks, emb_size = obs_reps.shape
        mask = mask.flatten()
        obs_reps = obs_reps.view(-1, emb_size)[mask]
        target_reps = target_reps.view(-1, emb_size)[mask]
        loss = self.mse(obs_reps, target_reps)
        return loss


@ObservationLoss.register("mse-with-length-normalization")
class RepresentationMSELossWithLengthNormalization(RepresentationMSELoss):
    def __init__(self) -> None:
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(
        self,
        obs_reps: torch.Tensor,  # (bs, seq*(na+no), hidden_size)
        target_reps: torch.Tensor,  # (bs, seq*(na+no), hidden_size)
        mask: torch.Tensor,  # (bs, seq*(na+no))
    ) -> torch.Tensor:
        return (
            (
                torch.nn.functional.mse_loss(
                    obs_reps,
                    target_reps,
                    reduction="none",
                ).sum(
                    -1
                )  # sum in the feature dimension
                * mask  # mask out PADs in the sequence dimension
            ).sum()  # Take the sum in sequence dim after masking
            / get_lengths_from_binary_sequence_mask(mask).to(
                dtype=obs_reps.dtype
            )  # divide by the length for each example in the batch to get mean
        ).mean()  # take the overall mean of all examples in the batch


class ActionLoss(torch.nn.Module, Registrable):
    def __init__(self, class_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.class_weight = class_weight

    def forward(
        self,
        logits: torch.FloatTensor,
        action_ids: torch.LongTensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        raise NotImplementedError


@ActionLoss.register("seq-ce-with-mask")
class SequenceCrossEntropyWithMask(ActionLoss):
    def forward(
        self,
        logits: torch.FloatTensor,
        action_ids: torch.LongTensor,
        mask: torch.BoolTensor,  # (bs, seq)
    ) -> torch.Tensor:
        loss = sequence_cross_entropy_with_logits(
            logits, action_ids, weights=mask, average="token", alpha=self.class_weight
        )

        return loss


class PlanningModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        beam_search: BeamSearch,
        action_loss: ActionLoss = SequenceCrossEntropyWithMask(),
        observation_loss: ObservationLoss = RepresentationMSELoss(),
        observation_loss_weight: float = 1.0,
        task_embedding: bool = False,
        top_k: int = 10,
        max_steps: int = 4,
        action_only: bool = False,
        predict_mode: bool = False,
        new_tokens: Optional[Dict[str, str]] = None,
        device_map: Optional[Dict[str, List[int]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self.action_loss = action_loss
        self.observation_loss = observation_loss
        self.observation_loss_weight = observation_loss_weight
        self.top_k = top_k
        self.action_only = action_only
        self.beam_search = beam_search
        self.sequence_accuracy = PlanningAccuracy([None, 3, 4])
        self.next_action_accuracy = CategoricalAccuracy()
        self.use_task = task_embedding
        self.success_rate = SuccessRate(
            [
                (1, None),
                (1, 3),
                (1, max_steps),
                (self.top_k, None),
                (self.top_k, 3),
                (self.top_k, max_steps),
            ]
        )
        self.miou = MeanIntersectionOverUnion(
            [
                (1, None),
                (1, 3),
                (1, max_steps),
                (self.top_k, None),
                (self.top_k, 3),
                (self.top_k, max_steps),
            ]
        )
        self.ed = EditDistance(
            [
                (1, None),
                (1, 3),
                (1, max_steps),
                (self.top_k, None),
                (self.top_k, 3),
                (self.top_k, max_steps),
            ]
        )
        self.action_loss_metric = Average()
        self.observation_loss_metric = Average()
        self.predict_mode = predict_mode
        self.new_tokens = new_tokens or {}
        end_action_text = "".join(
            [
                self.new_tokens.get("end_action_text", ""),
                self.new_tokens.get("end_of_action_text", ""),
            ]
        )
        self._end_action_idx = self.vocab.get_token_index(end_action_text, "actions")
        if self.predict_mode:
            self.prediction_buffer: Dict[str, Any] = {}

            # def reset_buffer(self, input, output) -> None:
            #    output.update(self.prediction_buffer)
            #    self.prediction_buffer = {}

            # self.register_forward_hook(reset_buffer)

    def add_to_prediction_buffer(self, key, value) -> None:
        if self.predict_mode:
            self.prediction_buffer[key] = value.detach().cpu()

    def reset_buffer(self, output) -> None:
        output.update(self.prediction_buffer)
        self.prediction_buffer = {}

    @property
    def num_actions(self) -> int:
        raise NotImplementedError

    @property
    def end_action_idx(self) -> int:
        """idx of the action that marks the end of action sequence.

        Raises:
            NotImplementedError: _description_

        Returns:
            int: _description_
        """
        raise NotImplementedError

    @property
    def pad_action_idx(self) -> int:
        """idx of the action that is used as PAD in the action sequence.

        Raises:
            NotImplementedError: _description_

        Returns:
            int: _description_
        """
        return self.vocab.get_token_index(self.vocab._padding_token)

    @property
    def num_obs_tokens(self) -> int:
        raise NotImplementedError

    def set_num_obs_tokens(self, observations: torch.Tensor) -> None:
        raise NotImplementedError

    def take_search_step(
        self,
        last_predictions: torch.LongTensor,  # (group, num_actions)
        state: Dict[str, torch.Tensor],
        step: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def compute_planning_metrics(
        self,
        beam_predictions: torch.Tensor,
        target: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> None:
        predictions, target = self.adjust_lengths(beam_predictions, target)
        self.success_rate(predictions, target)
        self.miou(predictions, target)
        self.sequence_accuracy(predictions, target)
        self.ed(predictions, target)
        self.next_action_accuracy(
            torch.nn.functional.one_hot(predictions[:, 0, 0], self.num_actions),
            target[:, 0],
        )

    def _get_metrics(self, reset: bool = False) -> Dict[str, float]:

        return {
            **self.get_planning_metrics(reset),
            "action_loss": self.action_loss_metric.get_metric(reset),
            "observation_loss": self.observation_loss_metric.get_metric(reset),
        }

    def get_planning_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            **self.success_rate.get_metric(reset),
            **self.miou.get_metric(reset),
            **self.sequence_accuracy.get_metric(reset),
            **self.ed.get_metric(reset),
            "next_action_accuracy": self.next_action_accuracy.get_metric(reset),
        }

    def get_extra_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {**self._get_metrics(reset), **self.get_extra_metrics(reset)}

    def adjust_lengths(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """If there is a length difference in predictions and gold, this method
        will add PADs to the shorter sequence.

        Additionally, it will also change all ids after the first occurrence
        of end_action_idx to pad_action_idx if not already in that form.

        Args:
            predictions (torch.Tensor): predictions of shape (batch_size, beam_size, seq_len)
            gold_labels (torch.Tensor): gold labels of shape (batch_size, seq_length)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: predictions and gold_labels after adjusting the lengths
        """
        # Replace extra end of seq or "do nothing" tokens with pad
        batch, beam_size, _seq_len = predictions.shape
        predictions = predictions.reshape(batch * beam_size, _seq_len)
        descending_numbers = torch.arange(
            predictions.shape[-1], 0, -1, device=predictions.device
        )  # create descending numbers
        end_of_actions = predictions == self.end_action_idx
        first_end_of_actions_index = torch.argmax(
            end_of_actions * descending_numbers, dim=-1
        )
        # A 0 in first_end_of_actions_index means we have hit the max_length
        # and no end of sequence was predicted. For that prediction,
        # we will take the last index as the do_nothing_index
        first_end_of_actions_index = torch.where(
            first_end_of_actions_index.bool(),
            first_end_of_actions_index,
            predictions.shape[-1] - 1,
        )
        predicted_seq_length = first_end_of_actions_index + 1
        # create mask with False where there should be PADs
        pad_assign_mask = get_mask_from_sequence_lengths(
            predicted_seq_length, max_length=predictions.shape[-1]
        )
        # Assign PAD action id to PAD positions in the prediction
        predictions = torch.where(pad_assign_mask, predictions, self.pad_action_idx)
        predictions = predictions.reshape(batch, beam_size, _seq_len)
        gold_length = gold_labels.shape[-1]
        predictions_length = predictions.shape[-1]
        if predictions_length > gold_length:
            # extend the gold with pads
            gold_labels = torch.cat(
                [
                    gold_labels,
                    torch.full(
                        (gold_labels.shape[0], predictions_length - gold_length),
                        self.pad_action_idx,
                        dtype=gold_labels.dtype,
                        device=gold_labels.device,
                    ),
                ],
                dim=-1,
            )
        elif predictions_length < gold_length:
            # extend predictions with pad
            predictions = torch.cat(
                [
                    predictions,
                    torch.full(
                        (*(predictions.shape[:-1]), gold_length - predictions_length),
                        self.pad_action_idx,
                        dtype=predictions.dtype,
                        device=predictions.device,
                    ),
                ],
                dim=-1,
            )
        # You can't create that mask here!
        # Each metric will handle length differences differently.
        # Create masks in the individual metrics.
        return predictions, gold_labels

    def forward_next_token_prediction(
        self,
        observations: torch.Tensor,
        actions: TextFieldTensors,
        task: Optional[TextFieldTensors] = None,
        unique_sequence_mask: Optional[torch.BoolTensor] = None,
        meta: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Performs next token prediction given past.

        This method is mainly used during training.

        Args:
            observations (torch.Tensor): _description_
            actions (TextFieldTensors): _description_
            task (Optional[TextFieldTensors], optional): _description_. Defaults to None.
            unique_sequence_mask (Optional[torch.BoolTensor], optional): _description_. Defaults to None.
            meta (Optional[Dict], optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            Dict[str, torch.Tensor]: Dict with loss and some other information.
        """
        raise NotImplementedError

    def forward_beam_search(
        self,
        prefix_observations: torch.Tensor,
        prefix_actions: TextFieldTensors,
        task: Optional[TextFieldTensors] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def setup_prediciton_buffer(
        self,
        observations: Optional[torch.Tensor] = None,
        actions: Optional[
            TextFieldTensors
        ] = None,  # shape (batch, traj_len, num_action_tokens)
        prefix_observations: Optional[torch.Tensor] = None,
        prefix_actions: Optional[TextFieldTensors] = None,
        target_observations: Optional[torch.Tensor] = None,
        target_actions: Optional[TextFieldTensors] = None,
        task: Optional[TextFieldTensors] = None,
        unique_sequence: Optional[torch.BoolTensor] = None,
        meta: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        return None

    def forward(
        self,
        observations: Optional[torch.Tensor] = None,
        actions: Optional[
            TextFieldTensors
        ] = None,  # shape (batch, traj_len, num_action_tokens)
        prefix_observations: Optional[torch.Tensor] = None,
        prefix_actions: Optional[TextFieldTensors] = None,
        target_observations: Optional[torch.Tensor] = None,
        target_actions: Optional[TextFieldTensors] = None,
        task: Optional[TextFieldTensors] = None,
        unique_sequence: Optional[torch.BoolTensor] = None,
        meta: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        results: Dict[str, torch.Tensor] = {}
        if observations is not None:
            self.set_num_obs_tokens(observations)
        elif prefix_observations is not None:
            self.set_num_obs_tokens(prefix_observations)

        if self.training:  # next action prediction
            assert observations is not None
            assert actions is not None
            results_ = self.forward_next_token_prediction(
                actions=actions,
                observations=observations,
                task=task,
                unique_sequence_mask=unique_sequence,
                meta=meta,
            )
            results.update(results_)

        if not self.training:
            self.setup_prediciton_buffer(
                observations=observations,
                actions=actions,
                prefix_actions=prefix_actions,
                prefix_observations=prefix_observations,
                target_observations=target_observations,
                target_actions=target_actions,
                task=task,
                unique_sequence=unique_sequence,
                meta=meta,
                **kwargs,
            )
            assert prefix_actions is not None
            assert prefix_observations is not None
            if self.predict_mode:
                results["prefix_actions"] = get_token_ids_from_text_field_tensors(
                    prefix_actions
                )
            all_top_k_predictions, log_probabilities = self.forward_beam_search(
                prefix_observations,
                prefix_actions,
                task,
            )
            # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
            # shape (log_probabilities): (batch_size, beam_size)
            target_action_indices = get_token_ids_from_text_field_tensors(
                target_actions
            )

            target_length_mask = get_text_field_mask(target_actions)
            if self.predict_mode:
                results["beam_predictions"] = all_top_k_predictions
                results["target_action_indices"] = target_action_indices
                results["target_length_mask"] = target_length_mask
            self.compute_planning_metrics(
                all_top_k_predictions, target_action_indices, target_length_mask
            )
        if self.predict_mode:
            self.reset_buffer(results)
        return results

    def convert_beams_to_action_strings(
        self, beams: torch.Tensor
    ) -> List[List[List[str]]]:
        return np.vectorize(
            lambda i: self.vocab.get_token_from_index(i, namespace="actions")
        )(beams.cpu().numpy()).tolist()

    def make_actions_human_readable(self, output_dict) -> Dict:
        out = {}
        out["predictions"] = self.convert_beams_to_action_strings(
            output_dict["beam_predictions"]
        )
        out["targets"] = self.convert_beams_to_action_strings(
            output_dict["target_action_indices"]
        )
        out["prefix_actions"] = self.convert_beams_to_action_strings(
            output_dict["prefix_actions"]
        )
        return out

    def preserve_raw_outputs(self, output_dict: Dict) -> Dict:
        return {
            "predictions_indices": output_dict["beam_predictions"],
            "targets_indices": output_dict["target_action_indices"],
            "prefix_actions_indices": output_dict["prefix_actions"],
        }

    def make_observations_human_readable(self, output_dict) -> Dict:
        raise NotImplementedError
