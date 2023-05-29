# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


"""Based on the 
[DecoderNet class](https://github.com/allenai/allennlp-models/blob/v2.9.3/allennlp_models/generation/modules/decoder_nets/decoder_net.py) 
from allennlp-models. The only difference is that we will drop the requirement of encoder output.
"""
from typing import Generic, Tuple, Optional, TypeVar, TypedDict
from allennlp.modules import Embedding
import torch
from allennlp.common import Registrable


class DecoderStateDict(TypedDict):
    pass


DecoderStateT = TypeVar("DecoderStateT", bound=DecoderStateDict)


class DecoderNetwork(torch.nn.Module, Registrable, Generic[DecoderStateT]):

    """
    This class abstracts the neural architectures for sequence modeling using a decoder only.

    """

    def __init__(
        self,
        decoding_dim: int,
        action_target_embedding_dim: int,
        observation_target_embedding_dim: int,
        action_embeddings: Embedding,
        decodes_parallel: bool = False,
        requires_all_previous_predictions: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            **kwargs,
        )
        self.action_target_embedding_dim = action_target_embedding_dim
        self.observation_target_embedding_dim = observation_target_embedding_dim
        self.decodes_parallel = decodes_parallel
        self.requires_all_previous_predictions = requires_all_previous_predictions
        self.action_embeddings = action_embeddings
        self.decoding_dim = decoding_dim
        self.action_projector = torch.nn.Linear(
            decoding_dim, self.action_embeddings.get_output_dim()
        )
        """
        Args:
            decoding_dim : `int`, required
                Defines dimensionality of output vectors.
            target_embedding_dim : `int`, required
                Defines dimensionality of target embeddings. Since this model takes a 
                processed version of it's output on a previous step, 
                (processed into indexes and back into index's embedding by a head on 
                top of this module) as input of following step, this is also an input dimensionality.
            decodes_parallel : `bool`, required
                Defines whether the decoder generates multiple next step predictions at 
                in a single `forward`. 
                This is will be set to true by masked self-attention based models like 
                transformer decoder but it will
                be set to false by RNN based decoders as they are truly autoregressive 
                and will always require a for loop
                for decoding multiple steps.
        """

    def get_output_dim(self) -> int:
        """
        Action output size
        """
        return self.get_actions_output_dim()

    def get_actions_output_dim(self) -> int:
        raise NotImplementedError

    def get_observations_output_dim(self) -> int:
        raise NotImplementedError

    def init_decoder_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> DecoderStateT:
        """
        Initialize the encoded state to be passed to the first decoding time step.
        This will be mostly a zero-init op.
        The exact specification of the keys and values of the `state` dict
        of the decoder network
        is left to the concrete implementation without any requirement
        of `state` definition compatibility across different implementations.

        Args:
            batch_size: Size of batch
        Returns:
            Dict[str, torch.Tensor]: Initial state
        """
        raise NotImplementedError()

    def reset_state(
        self, state: DecoderStateT, length_mask: torch.BoolTensor
    ) -> DecoderStateT:
        """Reset the state by picking the true (non-PAD) last state (for RNN based)
        or store the length_mask for transformer based decoder.

        Args:
            state (DecoderStateT): _description_
            length_mask (torch.BoolTensor): _description_

        Returns:
            DecoderStateT: _description_
        """
        raise NotImplementedError

    def forward(
        self,
        previous_state: DecoderStateT,
        action_previous_steps_predictions: torch.Tensor,
        observation_previous_steps_predictions: torch.Tensor,
        previous_steps_mask: Optional[
            torch.BoolTensor
        ] = None,  # (group or batch, steps_count)
    ) -> Tuple[DecoderStateT, torch.Tensor, torch.Tensor]:

        """
        Performs a decoding step, and returns dictionary with decoder hidden state or cache and the decoder output.

        Note:
            The decoder output is a 3d tensor (group_size, steps_count, decoding_dim)
            if `self.decodes_parallel` is True, else it is a 2d tensor with (group_size, decoding_dim).
            This is the most ugly part of the code.

        Args:
            action_previous_steps_predictions: Embeddings of actions predictions on previous steps including the current input.
                Shape: (group_size, steps_count+1, action_target_embedding_dim)
            observation_previous_steps_predictions: Embeddings of observation predictions on previous steps including the current input.
                Shape: (group_size, steps_count, observation_target_embedding_dim)
            previous_state : `StateT`, required
                previous state of decoder.
            previous_steps_mask: Length mask for PAD

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]:
            Tuple of new decoder state and decoder output of action and observation. Output should be used to generate out sequence elements
        """
        raise NotImplementedError()
