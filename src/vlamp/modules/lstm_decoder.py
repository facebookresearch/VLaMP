# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from typing import Tuple, Optional


import torch
from torch.nn import LSTMCell

from .decoder import DecoderNetwork, DecoderStateDict


class LstmCellDecoderDict(DecoderStateDict):
    decoder_context: torch.Tensor  # cell context (c) at a particular time
    decoder_hidden: torch.Tensor  # hidden state (h) of the cell at a particular time


@DecoderNetwork.register("lstm")
class LstmCellDecoderNetwrok(DecoderNetwork[LstmCellDecoderDict]):
    """
    This decoder net implements simple decoding network with unidirectional LSTMCell.
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        """
        Args:
            decoding_dim : `int`, required
                Defines dimensionality of output vectors.
            *_target_embedding_dim : `int`, required
                Defines dimensionality of input target embeddings.  Since this model takes it's output on a previous step
                as input of following step, this is also an input dimensionality.
        """

        super().__init__(
            decodes_parallel=False,
            requires_all_previous_predictions=False,
            **kwargs,
        )

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        # target_embedding_dim is a misnomer. It is the dimension of the input embedding at each step.
        self.target_embedding_dim = (
            self.action_target_embedding_dim + self.observation_target_embedding_dim
        )
        self._decoder_cell = LSTMCell(self.target_embedding_dim, self.decoding_dim)
        self.observation_projector = torch.nn.Linear(
            self.decoding_dim, self.observation_target_embedding_dim
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

    def encode_source(
        self, state: LstmCellDecoderDict, length_mask: torch.BoolTensor
    ) -> LstmCellDecoderDict:
        pass

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
        decoder_input = previous_steps_predictions[:, -1]
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
        # project the observation back to observation_target_dim
        return (
            {"decoder_hidden": decoder_hidden, "decoder_context": decoder_context},
            decoder_hidden,  # action shape (batch, decoding_dim)
            self.observation_projector(
                decoder_hidden
            ),  # obs shape (batch, observation_target_dim)
        )
