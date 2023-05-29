# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from typing import List, Tuple, Optional, Union
from allennlp.common import Registrable
from allennlp.modules import FeedForward
from allennlp.nn import Activation


import torch

from .decoder import DecoderNetwork, DecoderStateDict


class FeedForwardDecoderDict(DecoderStateDict):
    pass


class MLP(torch.nn.Module, Registrable):
    default_implementation: Optional[str] = "mlp"

    def __init__(self, ff: FeedForward) -> None:
        super().__init__()
        self.ff = ff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)

    @classmethod
    def create_feedforward(
        cls,
        input_dim: int,
        output_dim: int,
        num_hidden_layers: int = 1,
        hidden_dims: Optional[int] = None,
        hidden_activations: Optional[Activation] = None,
        dropout: Union[float, List[float]] = 0.0,
    ):
        if hidden_dims is None:
            hidden_dims = output_dim * 2
        if hidden_activations is None:
            hidden_activations = Activation.by_name("relu")()

        return cls(
            FeedForward(
                input_dim=input_dim,
                num_layers=num_hidden_layers + 1,
                hidden_dims=[hidden_dims] * num_hidden_layers + [output_dim],
                activations=[hidden_activations] * num_hidden_layers
                + [Activation.by_name("linear")()],
                dropout=dropout,
            )
        )


# register itself
MLP.register("mlp", "create_feedforward")(MLP)


@DecoderNetwork.register("ff")
class FeedForwardDecoderNetwrok(DecoderNetwork[FeedForwardDecoderDict]):
    """
    A simple stateless feedforward decoder
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
        input_dim = (
            self.action_target_embedding_dim + self.observation_target_embedding_dim
        )
        self.policy = MLP.create_feedforward(
            input_dim=input_dim, output_dim=self.decoding_dim
        )
        self.dynamics = MLP.create_feedforward(
            input_dim=input_dim, output_dim=self.observation_target_embedding_dim
        )

    def init_decoder_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> FeedForwardDecoderDict:
        # TODO: we probably want to try learned initial states.
        return {}

    def get_actions_output_dim(self) -> int:
        return self.decoding_dim

    def get_observations_output_dim(self) -> int:
        return self.observation_target_embedding_dim

    def forward(
        self,
        previous_state: FeedForwardDecoderDict,
        action_previous_steps_predictions: torch.Tensor,  # shape: (batch or group, num_previous_steps+1, action_target_dim)
        observations_previous_steps_predictions: torch.Tensor,  # shape: (batch or group, num_previous_steps+1, observation_target_dim)
        previous_steps_mask: Optional[torch.BoolTensor] = None,  # PAD mask on source
    ) -> Tuple[FeedForwardDecoderDict, torch.Tensor, torch.Tensor]:

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
        decoder_hidden = self.policy(decoder_input)

        # Predict the observation
        predicted_obs = self.dynamics(
            torch.cat(
                [
                    self.action_projector(decoder_hidden),
                    observations_previous_steps_predictions[:, -1, :],
                ],
                dim=-1,
            )
        )
        return (
            {},
            decoder_hidden,  # action shape (batch, decoding_dim)
            predicted_obs,  # obs shape (batch, observation_target_dim)
        )


@DecoderNetwork.register("action-only-ff")
class ActionOnlyFeedForwardDecoderNetwrok(DecoderNetwork[FeedForwardDecoderDict]):
    """
    A simple stateless feedforward decoder that uses only actions.
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
        input_dim = self.action_target_embedding_dim
        self.policy = MLP.create_feedforward(
            input_dim=input_dim, output_dim=self.decoding_dim
        )

    def init_decoder_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> FeedForwardDecoderDict:
        # TODO: we probably want to try learned initial states.
        return {}

    def get_actions_output_dim(self) -> int:
        return self.decoding_dim

    def get_observations_output_dim(self) -> int:
        return self.observation_target_embedding_dim

    def forward(
        self,
        previous_state: FeedForwardDecoderDict,
        action_previous_steps_predictions: torch.Tensor,  # shape: (batch or group, num_previous_steps+1, action_target_dim)
        observations_previous_steps_predictions: torch.Tensor,  # shape: (batch or group, num_previous_steps+1, observation_target_dim)
        previous_steps_mask: Optional[torch.BoolTensor] = None,  # PAD mask on source
    ) -> Tuple[FeedForwardDecoderDict, torch.Tensor, torch.Tensor]:

        decoder_input = action_previous_steps_predictions[:, -1, :]
        decoder_hidden = self.policy(decoder_input)

        # Predict the observation
        predicted_obs = observations_previous_steps_predictions[:, -1, :]
        return (
            {},
            decoder_hidden,  # action shape (batch, decoding_dim)
            predicted_obs,  # obs shape (batch, observation_target_dim)
        )
