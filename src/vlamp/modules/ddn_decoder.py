# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from typing import Tuple, Optional


import torch
from torch.nn import LSTMCell

from .decoder import DecoderNetwork, DecoderStateDict


class LstmCellDecoderDict(DecoderStateDict):
    decoder_context: torch.Tensor  # cell context (c) at a particular time
    decoder_hidden: torch.Tensor  # hidden state (h) of the cell at a particular time


@DecoderNetwork.register("ddn")
class DDNDecoderNetwrok(DecoderNetwork[LstmCellDecoderDict]):
    """
    Implements the core decoding step of the DDN paper.
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            decodes_parallel=False,
            requires_all_previous_predictions=False,
            **kwargs,
        )
        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # TODO: Support multiple layers of LSTMCell
        # target_embedding_dim is a misnomer. It is the dimension of the input embedding at each step.
        self.input_dim = (
            self.action_target_embedding_dim + self.observation_target_embedding_dim
        )
        # This is the conjugate dynamics or the policy model
        self._decoder_cell = LSTMCell(self.input_dim, self.decoding_dim)

        # This is the "dynamics model"
        self.observation_projector = torch.nn.Linear(
            self.input_dim, self.observation_target_embedding_dim
        )

    def init_decoder_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LstmCellDecoderDict:
        # TODO: we probably want to try learned initial states.
        return {
            "decoder_hidden": torch.zeros(
                batch_size,
                self.decoding_dim,
                device=device,
                dtype=dtype,
            ),  # shape: (batch_size, decoder_output_dim)
            "decoder_context": torch.zeros(
                batch_size, self.decoding_dim, device=device, dtype=dtype
            )
            #                  shape: (batch_size, decoder_output_dim)
        }

    def get_actions_output_dim(self) -> int:
        return self.decoding_dim

    def get_observations_output_dim(self) -> int:
        return self.observation_target_embedding_dim

    def forward(
        self,
        previous_state: LstmCellDecoderDict,
        action_previous_steps_predictions: torch.Tensor,  # shape: (batch or group, num_previous_steps+1, action_target_dim)
        observations_previous_steps_predictions: torch.Tensor,  # shape: (batch or group, num_previous_steps+1, observation_target_dim)
        previous_steps_mask: Optional[torch.BoolTensor] = None,  # PAD mask on source
    ) -> Tuple[LstmCellDecoderDict, torch.Tensor, torch.Tensor]:

        decoder_hidden = previous_state["decoder_hidden"]
        decoder_context = previous_state["decoder_context"]
        # ISSUE: This cat is expensive. Do cat after indexing
        previous_steps_predictions = torch.cat(
            [
                action_previous_steps_predictions,
                observations_previous_steps_predictions,
            ],
            dim=-1,
        )
        # TODO: Make this condition on the shape. We will only pass last step predictions
        # to decoders that only require that.
        # shape: (batch or group, target_embedding_dim)
        decoder_input = previous_steps_predictions[:, -1, :]
        previous_hidden, previous_context = (
            previous_state["decoder_hidden"],
            previous_state["decoder_context"],
        )
        decoder_hidden, decoder_context = self._decoder_cell(
            decoder_input, (decoder_hidden, decoder_context)
        )

        # Update state only if current token is not PAD
        # sequence: t1 t2 PAD PAD
        # mask: T T F F
        # seq_length: 2
        if previous_steps_mask is not None:
            assert (
                previous_steps_mask.shape[-1]
                == action_previous_steps_predictions.shape[-2]
            )
            update_mask = previous_steps_mask[..., -1].view(
                -1, 1
            )  # (group or batch, 1)
            decoder_hidden = torch.where(update_mask, decoder_hidden, previous_hidden)
            decoder_context = torch.where(
                update_mask, decoder_context, previous_context
            )
        # return the entire hidden as the representation for action
        # Predict the observation
        predicted_obs = self.observation_projector(
            torch.cat(
                [
                    self.action_projector(decoder_hidden),
                    observations_previous_steps_predictions[:, -1, :],
                ],
                dim=-1,
            )
        )
        return (
            {"decoder_hidden": decoder_hidden, "decoder_context": decoder_context},
            decoder_hidden,  # action shape (batch, decoding_dim)
            predicted_obs,  # obs shape (batch, observation_target_dim)
        )


@DecoderNetwork.register("action-only-ddn")
class ActionOnlyDDNDecoderNetwrok(DecoderNetwork[LstmCellDecoderDict]):
    """
    Only uses actions and ignores observations
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            decodes_parallel=False,
            requires_all_previous_predictions=False,
            **kwargs,
        )
        self.input_dim = self.action_target_embedding_dim
        # This is the conjugate dynamics or the policy model
        self._decoder_cell = LSTMCell(self.input_dim, self.decoding_dim)

    def init_decoder_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LstmCellDecoderDict:
        # TODO: we probably want to try learned initial states.
        return {
            "decoder_hidden": torch.zeros(
                batch_size,
                self.decoding_dim,
                device=device,
                dtype=dtype,
            ),  # shape: (batch_size, decoder_output_dim)
            "decoder_context": torch.zeros(
                batch_size, self.decoding_dim, device=device, dtype=dtype
            )
            #                  shape: (batch_size, decoder_output_dim)
        }

    def get_actions_output_dim(self) -> int:
        return self.decoding_dim

    def get_observations_output_dim(self) -> int:
        return self.observation_target_embedding_dim

    def forward(
        self,
        previous_state: LstmCellDecoderDict,
        action_previous_steps_predictions: torch.Tensor,  # shape: (batch or group, num_previous_steps+1, action_target_dim)
        observations_previous_steps_predictions: torch.Tensor,  # shape: (batch or group, num_previous_steps+1, observation_target_dim)
        previous_steps_mask: Optional[torch.BoolTensor] = None,  # PAD mask on source
    ) -> Tuple[LstmCellDecoderDict, torch.Tensor, torch.Tensor]:

        decoder_hidden = previous_state["decoder_hidden"]
        decoder_context = previous_state["decoder_context"]
        # ISSUE: This cat is expensive. Do cat after indexing
        decoder_input = action_previous_steps_predictions[:, -1, :]
        previous_hidden, previous_context = (
            previous_state["decoder_hidden"],
            previous_state["decoder_context"],
        )
        decoder_hidden, decoder_context = self._decoder_cell(
            decoder_input, (decoder_hidden, decoder_context)
        )

        # Update state only if current token is not PAD
        # sequence: t1 t2 PAD PAD
        # mask: T T F F
        # seq_length: 2
        if previous_steps_mask is not None:
            assert (
                previous_steps_mask.shape[-1]
                == action_previous_steps_predictions.shape[-2]
            )
            update_mask = previous_steps_mask[..., -1].view(
                -1, 1
            )  # (group or batch, 1)
            decoder_hidden = torch.where(update_mask, decoder_hidden, previous_hidden)
            decoder_context = torch.where(
                update_mask, decoder_context, previous_context
            )
        predicted_obs = observations_previous_steps_predictions[:, -1, :]
        
        return (
            {"decoder_hidden": decoder_hidden, "decoder_context": decoder_context},
            decoder_hidden,  # action shape (batch, decoding_dim)
            predicted_obs,  # obs shape (batch, observation_target_dim)
        )
