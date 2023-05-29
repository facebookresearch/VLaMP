# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from typing import Dict, List, Optional, Tuple, Union

from vlamp.modules.observation_encoders.frame_encoders import (
    ObservationFeaturesEncoder,
)
from .planning_base_model import PlanningModel
import torch
import torch.nn.functional as F
from allennlp.common import Lazy
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.modules import Embedding
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.util import (
    get_text_field_mask,
    get_token_ids_from_text_field_tensors,
)
from vlamp.modules.decoder import DecoderNetwork
from vlamp.flags import DEBUG
import numpy as np


class TiedEmbeddingScorer(torch.nn.Module):
    def __init__(
        self,
        embedding: Embedding,
        projector: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.projector = projector
        self.embedding = embedding

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp = self.project_into_action_embedding_space(inp)
        return F.linear(inp, self.embedding.weight, bias=None)

    def project_into_action_embedding_space(self, inp: torch.Tensor) -> torch.Tensor:
        return self.projector(inp)

    def get_projector(self) -> torch.nn.Module:
        return self.projector if self.projector is not None else torch.nn.Identity()


# TODO: Create a special constructor to construct the lazy objects
class ProceduralPlanningModel(PlanningModel):
    default_implementation: Optional[str] = "sequence-model"
    state_all_previous_action_indices_key = "previous_steps_actions"
    state_all_previous_observation_embeddings_key = "previous_steps_observations"
    state_all_previous_mask_key = "previous_steps_mask"
    state_last_observation_embeddings_key = "last_observation"
    state_last_mask_key = "last_mask"

    def __init__(
        self,
        vocab: Vocabulary,
        beam_search: BeamSearch,
        action_embedder: Embedding,
        observations_encoder: ObservationFeaturesEncoder,  # TODO: Generalize to consume video frames
        decoder: DecoderNetwork,
        source_sampling_ratio: float = 0.0,
        predict_mode: bool = False,
        **kwargs,
    ) -> None:
        """

        Args:
            vocab (Vocabulary): _description_
            decoder (DecoderOnlyNetwork): _description_
            source_sampling_ratio (float, optional): % source taken from ground truth =
                (1 - source_sampling_ratio)*100.
                Defaults to 0.0, meaning complete teacher forcing with all inputs from ground truth.
        """
        super().__init__(
            vocab, beam_search=beam_search, predict_mode=predict_mode, **kwargs
        )
        self.source_sampling_ratio = source_sampling_ratio
        self.action_embedder = action_embedder
        self.observations_encoder = observations_encoder
        self._num_actions = vocab.get_vocab_size(namespace="actions")
        self.decoder = decoder
        self.action_logit_projector = TiedEmbeddingScorer(
            self.action_embedder, projector=self.decoder.action_projector
        )

    def set_num_obs_tokens(self, observations: torch.Tensor) -> None:
        pass

    @property
    def num_actions(self) -> int:
        return self.vocab.get_vocab_size("actions")

    def single_pass_decode(
        self,
        source_action_indices,
        source_observation_embeddings,
        source_mask=None,
        state=None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Note:
            Updates the provided state dict.
        """
        source_action_embeddings = self.action_embedder(source_action_indices)
        if state is None:
            state = self.decoder.init_decoder_state(
                source_action_indices,
                source_observation_embeddings,
            )  # shape: Dict of type StateT for that decoder
        new_state, action_decoder_output, observation_decoder_output = self.decoder(
            state,
            source_action_embeddings,
            source_observation_embeddings,
            previous_steps_mask=source_mask,
        )
        state.update(new_state)
        action_logits = self.action_logit_projector(
            action_decoder_output
        )  # (batch, seq_len -1, total_actions)
        predicted_observations = observation_decoder_output
        return state, action_logits, predicted_observations

    def create_source_and_target_from_complete_sequence(
        self,
        observations: torch.Tensor,
        actions: TextFieldTensors,
        task: Optional[TextFieldTensors] = None,
        unique_sequence_mask: Optional[torch.BoolTensor] = None,
    ):

        action_indices = get_token_ids_from_text_field_tensors(
            actions
        )  # (batch, seq_len)

        length_mask: torch.BoolTensor = get_text_field_mask(
            actions
        )  # (batch, seq_len) masks out the PADs
        target_action_indices = (
            action_indices[..., 1:]
        ).contiguous()  # (batch, seq_len - 1)
        target_length_mask: torch.BoolTensor = (length_mask[..., 1:]).contiguous()  # type: ignore
        source_length_mask: torch.BoolTensor = (length_mask[..., :-1]).contiguous()  # type: ignore
        source_action_indices = action_indices[..., :-1]

        # Prepare observations
        observations_embeddings = self.observations_encoder(
            observations
        )  # (batch, seq_len, observation_emb_dim)
        target_observation_embeddings = observations_embeddings[..., 1:, :]
        source_observation_embeddings = observations_embeddings[..., :-1, :]
        if unique_sequence_mask is not None:
            action_indices = action_indices[unique_sequence_mask]
            target_action_indices = target_action_indices[unique_sequence_mask]
            length_mask = length_mask[unique_sequence_mask]  # type: ignore
            target_length_mask = target_length_mask[unique_sequence_mask]  # type: ignore
            source_length_mask = source_length_mask[unique_sequence_mask]  # type: ignore
            source_action_indices = source_action_indices[unique_sequence_mask]
            source_observation_embeddings = source_observation_embeddings[
                unique_sequence_mask
            ]
            target_observation_embeddings = target_observation_embeddings[
                unique_sequence_mask
            ]
        return (
            source_action_indices,
            source_observation_embeddings,
            source_length_mask,
            target_action_indices,
            target_observation_embeddings,
            target_length_mask,
        )

    def forward_next_token_prediction(
        self,
        observations: torch.Tensor,
        actions: TextFieldTensors,
        task: Optional[TextFieldTensors] = None,
        unique_sequence_mask: Optional[torch.BoolTensor] = None,
        meta: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        # num_steps = number of actions in the trajectory
        # T = num_steps + 1 is the total observations available for the trajectory
        # Hence we have the data in the form o_1, a_1, o_2, a_2, ..., o_T-1, a_T-1, o_T.
        # We expect the dataset reader to send data in the following form:
        # actions:
        #   [a_e, a_1, ..., a_T-2, a_T-1] if extra end observation is not enabled (length=T) and
        #   [a_e, a_1, ..., a_T-2, a_T-1, a_e] if extra end observation is enabled (length=T+1)
        # observations:
        #   [o_1, ..., o_T-1, o_T] if extra end observation is not enabled (length=T) and
        #   [o_1, ..., o_T-1, o_T, o_T] if extra end observation is enabled (length=T+1).
        #   Note that in the latter case the last observation is repeated.
        # Here, a_e is a special "do nothing" action.
        # We will form the source and target as follows:
        # source:
        #   1. [(a_e, o_1), (a_1, o_2), ..., (a_T-2, o_T-1)] (len=T-1)
        #       if extra end obs. is not enabled
        #   2. [(a_e, o_1), (a_1, o_2), ..., (a_T-1, o_T)] (len=T)
        #       if extra end obs. is enabled
        # target:
        #   1. [(a_1, o_2), ..., (a_T-1, o_T)] (len=T-1)
        #       if extra end obs. is not enabled
        #   2. [(a_1, o_2), ..., (a_T-1, o_T), (a_e, o_T)] (len=T)
        #       if extra end obs. is enabled
        # In general, we will call seq_len to be the length of actions, which can be T or T+1.

        (
            source_action_indices,
            source_observation_embeddings,
            source_length_mask,
            target_action_indices,
            target_observation_embeddings,
            target_length_mask,
        ) = self.create_source_and_target_from_complete_sequence(
            observations=observations,
            actions=actions,
            task=task,
            unique_sequence_mask=unique_sequence_mask,
        )
        results: Dict[str, torch.Tensor] = {}
        # compute loss if we have targets
        # ISSUE: even during true prediction mode,
        #   if adding extra observations is on, the seq_len will be 2.
        #   To fix this we need to share a flag between the data reading pipeline and the model.
        # attempt single pass decoding if the decoder supports it and we are
        # using full teacher forcing.
        if target_action_indices.numel() > 0:
            (
                loss,
                next_actions_logits,
                next_observations,
            ) = self.compute_loss_using_next_action_prediction(
                source_action_indices,
                source_observation_embeddings,
                target_action_indices,
                target_observation_embeddings,
                target_length_mask,
            )
            results["loss"] = loss
            results["next_action_predictions"] = torch.max(next_actions_logits, -1)[1]

        else:
            results["loss"] = torch.tensor(
                0.0, device=observations.device, dtype=observations.dtype
            )
            results["next_action_predictions"] = torch.empty_like(target_action_indices)

        return results

    def prepare_prefix(
        self, actions: TextFieldTensors, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rearrange and split the prefix.

        We expect the prefix to have the last input prepended instead of appended.
        So a batch sequence:
        a11, a12, a13, pad, pad
        a21, a22, a23, a24, a25
        would come int as
        a13, a11, a12, pad, pad
        a25, a21, a22, a23, a24

        We will separate out the last tokens
        """
        action_indices = get_token_ids_from_text_field_tensors(actions)
        mask = get_text_field_mask(actions)
        prefix_action_indices = action_indices[..., 1:]
        observations_emb = self.observations_encoder(observations)
        prefix_observations = observations_emb[..., 1:, :]
        start_actions = action_indices[..., 0]
        start_observations = observations_emb[..., 0, :]
        return (
            prefix_action_indices,
            prefix_observations,
            mask[..., 1:],
            start_actions,
            start_observations,
        )

    def forward_beam_search(
        self,
        prefix_observations,
        prefix_actions,
        task: Optional[TextFieldTensors] = None,
    ):
        assert prefix_observations is not None
        # prepare the decoder for inference
        # by first encoding the prefix
        (
            prefix_action_indices,
            prefix_observations,
            prefix_mask,
            start_actions,
            start_observations,
        ) = self.prepare_prefix(prefix_actions, prefix_observations)
        (
            source_state,
            source_predicted_actions,
            source_predicted_observations,
        ) = self.encode_source(prefix_action_indices, prefix_observations, prefix_mask)
        all_top_k_predictions, log_probabilities = self._search(
            start_actions, start_observations, state=source_state
        )

        return all_top_k_predictions, log_probabilities

    def encode_source(
        self,
        source_action_indices,
        source_observation_embeddings,
        source_mask,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        # Since we are encoding the source, we don't have any past state right now.
        # For short examples we might get empty prefix.
        # In that case, just initialize the decoder state
        if source_action_indices.shape[-1] == 0:
            batch = source_action_indices.shape[0]
            num_actions = self.vocab.get_vocab_size("actions")
            return (
                self.init_decoder_state(
                    source_action_indices=source_action_indices,
                    source_observation_embeddings=source_observation_embeddings,
                ),
                torch.empty(
                    batch,
                    0,
                    num_actions,
                    device=source_action_indices.device,
                    dtype=source_observation_embeddings.dtype,
                ),
                torch.empty_like(source_observation_embeddings),
            )

        # Let the two decoding methods create a state.
        if self.decoder.decodes_parallel:
            new_state, action_logits, predicted_observations = self.single_pass_decode(
                source_action_indices=source_action_indices,
                source_observation_embeddings=source_observation_embeddings,
                source_mask=source_mask,
            )
        else:
            new_state, action_logits, predicted_observations = self.loop_decode(
                source_action_indices=source_action_indices,
                source_observation_embeddings=source_observation_embeddings,
                source_mask=source_mask,
            )

        return new_state, action_logits, predicted_observations

    def init_decoder_state(
        self,
        source_action_indices: torch.Tensor,
        source_observation_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size: int = source_action_indices.shape[0]
        device: torch.device = source_action_indices.device
        dtype: torch.dtype = source_observation_embeddings.dtype
        decoder_state = self.decoder.init_decoder_state(batch_size, device, dtype=dtype)
        return decoder_state

    def update_state(
        self,
        state: Dict[str, torch.Tensor],
        last_action_indices: torch.Tensor,  # shape: (group,)
        last_observation_embeddings: torch.Tensor,  # shape: (group, obs_emb_dim)
        last_source_mask: Optional[torch.BoolTensor] = None,
    ) -> None:
        """If previous history exists, append last action and observation to it. If
        it does not exist, create it.

        Args:
            state (Dict[str, torch.Tensor]): _description_
            last_action_indices (torch.Tensor): _description_
        """
        if self.state_all_previous_action_indices_key in state:
            state[self.state_all_previous_action_indices_key] = torch.cat(
                [
                    state[self.state_all_previous_action_indices_key],
                    last_action_indices.unsqueeze(-1),
                ],
                dim=-1,
            )
            assert self.state_all_previous_observation_embeddings_key in state
            state[self.state_all_previous_observation_embeddings_key] = torch.cat(
                [
                    state[self.state_all_previous_observation_embeddings_key],
                    last_observation_embeddings.unsqueeze(-2),
                ],
                dim=-2,
            )
            if last_source_mask is not None:
                assert self.state_all_previous_mask_key in state
                state[self.state_all_previous_mask_key] = torch.cat(
                    [
                        state[self.state_all_previous_mask_key],
                        last_source_mask.unsqueeze(-1),
                    ],
                    dim=-1,
                )
            else:
                assert self.state_all_previous_mask_key not in state
        else:
            state[
                self.state_all_previous_action_indices_key
            ] = last_action_indices.unsqueeze(-1)
            assert self.state_all_previous_observation_embeddings_key not in state
            state[
                self.state_all_previous_observation_embeddings_key
            ] = last_observation_embeddings.unsqueeze(-2)
            if last_source_mask is not None:
                assert self.state_all_previous_mask_key not in state
                state[self.state_all_previous_mask_key] = last_source_mask.unsqueeze(-1)

    def predict_next_logits(
        self,
        last_step_action_indices: torch.Tensor,
        last_step_observation_embeddings: torch.Tensor,
        state: Dict[str, torch.Tensor],
        last_step_source_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Assumes state is not updated, updates the state and then uses entire
        previous history to predict the logits of next step and observation embeddings.

        Note:
            This method requires a state dict and it returns a
            new state dict with just the new decoder state. So
            it is the caller's responsibility to incorporate
            the new decoder state into the overall state if it wants to.

        Args:
            last_step_action_indices (torch.Tensor): _description_
            last_step_observation_embeddings (torch.Tensor): _description_
            state (Dict[str, torch.Tensor]): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]: predicted action logits, observation embeddings and state
        """
        # Add the last steps to previous history
        # TODO: These concats can be expensive and might not be required by
        self.update_state(
            state,
            last_step_action_indices,
            last_step_observation_embeddings,
            last_source_mask=last_step_source_mask,
        )
        (
            new_decoder_state,
            action_decoder_output,  # (group, steps, action_decoder_dim)
            observation_decoder_output,  # (group, steps, obs_emb_dim)
        ) = self.decoder(
            state,
            self.action_embedder(state[self.state_all_previous_action_indices_key]),
            state[self.state_all_previous_observation_embeddings_key],
            previous_steps_mask=state.get(self.state_all_previous_mask_key),
        )

        if self.decoder.decodes_parallel:
            predicted_action_logits = self.action_logit_projector(
                action_decoder_output[..., -1, :]
            )
        else:
            predicted_action_logits = self.action_logit_projector(action_decoder_output)

        return predicted_action_logits, observation_decoder_output, new_decoder_state

    def take_search_step(
        self,
        last_step_action_indices: torch.Tensor,
        state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """For using with beam search.
        The state should only have decoder initial state and the first observation embeddings.
        """
        step_observation_embeddings = state[self.state_last_observation_embeddings_key]
        step_mask: torch.BoolTensor = torch.ones(  # type: ignore
            step_observation_embeddings.shape[0],
            device=step_observation_embeddings.device,
            dtype=torch.bool,
        )
        action_logits, predicted_observations, new_state = self.predict_next_logits(
            last_step_action_indices,
            step_observation_embeddings,
            state,
            last_step_source_mask=step_mask,
        )
        state.update(new_state)
        action_probabilities = torch.nn.functional.log_softmax(action_logits, dim=-1)
        state[self.state_last_observation_embeddings_key] = predicted_observations
        return action_probabilities, state

    def _search(
        self,
        start_action_indices: torch.Tensor,
        start_observation_embeddings: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor]:
        """
        This method will be called in the following two situations:

        1. Predicting at inference time with no prior state
        2. Predicting at inference time with prefix encoded and stored in the state.
        """
        if state is None:
            state = self.init_decoder_state(
                start_action_indices, start_observation_embeddings
            )
        # Since the beam search interface only allows step function
        # with one last_step argument that is a tensor, we keep
        # more information in the state dict.
        # In particular, we use the one argument for action logits and
        # store last observation in the state.
        state[self.state_last_observation_embeddings_key] = start_observation_embeddings
        all_top_k_predictions, log_probabilities = self.beam_search.search(
            start_action_indices, state, self.take_search_step
        )
        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        return all_top_k_predictions, log_probabilities

    def compute_loss_using_next_action_prediction(
        self,
        source_action_indices,
        source_observation_embeddings,
        target_action_indices,
        target_observation_embeddings,
        target_length_mask,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_logits: torch.Tensor  # complete sequence of predicted
        # action logits of shape (batch, seq_len -1, total_actions)
        predicted_observations: torch.Tensor  # predicted observation embeddings (batch, seq_len-1, obs_emb_dim)
        # shape: Dict of type StateT for that decoder
        results: Dict[str, torch.Tensor] = {}

        if self.source_sampling_ratio == 0 and self.decoder.decodes_parallel:
            _, action_logits, predicted_observations = self.single_pass_decode(
                source_action_indices,
                source_observation_embeddings,
            )
        else:  # perform predictions in loop
            _, action_logits, predicted_observations = self.loop_decode(
                source_action_indices, source_observation_embeddings
            )

            # TODO: Allow to drop the last predict of "do nothing"
            results["action_logits"] = action_logits
            results["predicted_observations"] = predicted_observations

        # Compute loss
        # TODO: move the loss computations into their own classes.
        action_loss = self.action_loss(
            action_logits,  # type: ignore
            target_action_indices,  # type: ignore
            target_length_mask,
        )
        self.action_loss_metric(action_loss.item())
        if self.observation_loss_weight > 0:
            observation_loss = self.observation_loss(
                predicted_observations,
                target_observation_embeddings,
                target_length_mask,
            )
            results["loss"] = (
                action_loss + self.observation_loss_weight * observation_loss
            )
            self.observation_loss_metric(observation_loss.item())
        else:
            results["loss"] = action_loss
            self.observation_loss_metric(0)
        # TODO: This is ok as long as compute_loss is called only once in one step.
        return (
            results["loss"],
            action_logits,
            predicted_observations,
        )

    def loop_decode(
        self,
        source_action_indices,
        source_observation_embeddings,
        source_mask=None,
        state=None,
    ):
        """
        Following are the scenarios when this method will be use:

        1. Encoding the prefix during inference when the decoder does
            not support parallel decode. The predictions are done through the step method.
        2. Performing next step only predictions.

        We want to tell the state about the true source lengths.
        Through `source_mask`, this method knows about the true source lengths.
        But it is the decoder that knows how to best return the next state based
        on this information so we will just pass the source_mask to the decoder.

        Note:
            Updates the provided state.
        """
        batch, num_decoding_steps = source_action_indices.shape

        #                                           #
        # variables that change during the loops    #
        #                                           #
        current_action_index: torch.Tensor = source_action_indices[
            ..., 0
        ]  # shape: (batch,)
        current_observation_embedding: torch.Tensor = source_observation_embeddings[
            ..., 0, :
        ]  # shape (batch, obs_emb_dim) # shape: (batch, obs_emb_dim)
        if source_mask is not None:
            current_source_mask: Optional[torch.Tensor] = source_mask[..., 0]
        else:
            current_source_mask = None
        action_logits_list: List[
            torch.Tensor
        ] = []  # predicted logits of each time step so far
        predicted_observations_list: List[
            torch.Tensor
        ] = []  # predicted observation embeddings so far
        if state is None:
            state = self.init_decoder_state(
                source_action_indices=source_action_indices,
                source_observation_embeddings=source_observation_embeddings,
            )
        else:  # while predicting we will already have a state due to source encoding
            pass

        #               #
        # start looping #
        #               #
        for decoding_step in range(num_decoding_steps):
            (
                predicted_action_logits,
                predicted_observation,
                new_state,
            ) = self.predict_next_logits(
                current_action_index,
                current_observation_embedding,
                state,
                last_step_source_mask=current_source_mask,
            )
            state.update(new_state)

            action_logits_list.append(predicted_action_logits)
            predicted_observations_list.append(predicted_observation)

            # Update the current_* using the decoder outputs
            if decoding_step == num_decoding_steps - 1:
                break  # done
            take_prediction = (
                self.training and torch.rand(1).item() < self.source_sampling_ratio
            )
            current_action_index = (
                predicted_action_logits.max(-1)[1]
                if take_prediction
                else source_action_indices[..., decoding_step + 1]
            )  # take predicted action or ground truth next action as next input
            # If not training, we will do beam search which is not done here.
            # TODO: Provide a way to sample during validation?

            current_observation_embedding = (
                predicted_observation
                if take_prediction
                else source_observation_embeddings[..., decoding_step + 1, :]
            )
            if source_mask is not None:
                current_source_mask = source_mask[..., decoding_step + 1]
        # Consolidate the predictions
        action_logits = torch.stack(
            action_logits_list, dim=-2
        )  # (batch, seq_len -1, total_actions)
        predicted_observations = torch.stack(
            predicted_observations_list, dim=-2
        )  # (batch, seq_len-1, total_actions)
        return state, action_logits, predicted_observations

    def convert_beams_to_action_strings(
        self, beams: torch.Tensor
    ) -> List[List[List[str]]]:
        return np.vectorize(
            lambda i: self.vocab.get_token_from_index(i, namespace="actions")
        )(beams.numpy()).tolist()

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[torch.Tensor, List]]:
        for k, v in output_dict.items():
            if isinstance(v, torch.Tensor):
                output_dict[k] = v.detach().cpu()
        out = self.make_actions_human_readable(output_dict)
        output_dict.update(out)
        batch_size = len(output_dict["predictions"])
        prefix_length = int(
            output_dict["prefix_length_with_pads"][0]
        )  # All entries are the same
        output_dict["predicted_observations"] = torch.cat(
            [
                output_dict["previous_steps_observations"],
                output_dict["last_observation"].unsqueeze(1),
            ],
            dim=1,
        )[:, prefix_length:, :]

        # split batch and beam dimensions
        output_dict["previous_steps_observations"] = output_dict[
            "previous_steps_observations"
        ].reshape(
            batch_size, -1, *(output_dict["previous_steps_observations"].shape[-2:])
        )
        output_dict["predicted_observations"] = output_dict[
            "predicted_observations"
        ].reshape(batch_size, -1, *(output_dict["predicted_observations"].shape[-2:]))
        output_dict["previous_steps_actions"] = output_dict[
            "previous_steps_actions"
        ].reshape(batch_size, -1, output_dict["previous_steps_actions"].shape[-1])
        output_dict["last_observation"] = output_dict["last_observation"].reshape(
            batch_size, -1, output_dict["last_observation"].shape[-1]
        )
        output_dict["previous_steps_mask"] = output_dict["previous_steps_mask"].reshape(
            batch_size, -1, output_dict["previous_steps_mask"].shape[-1]
        )
        return output_dict

    @property
    def end_action_idx(self) -> int:
        return self._end_action_idx

    @property
    def pad_action_idx(self) -> int:
        return self.vocab.get_token_index(self.vocab._padding_token)

    @classmethod
    def construct_decoder(
        cls,
        vocab: Vocabulary,
        decoder: Lazy[DecoderNetwork],
        action_embedder: Embedding,
        observations_encoder: ObservationFeaturesEncoder,  # TODO: Generalize to consume video frames
        beam_search: Lazy[BeamSearch] = Lazy(BeamSearch),
        source_sampling_ratio: float = 0.0,
        top_k: int = 10,
        max_steps: int = 4,
        min_steps: int = 1,
        predict_mode: bool = False,
        **kwargs,
    ) -> "ProceduralPlanningModel":
        """
        See the constructor for the arguments.
        """
        decoder = decoder.construct(
            action_embeddings=action_embedder,
            action_target_embedding_dim=action_embedder.get_output_dim(),
            observation_target_embedding_dim=observations_encoder.get_output_dim(),
        )
        beam_search_ = beam_search.construct(
            vocab=vocab,
            end_index=vocab.get_token_index("do nothing", namespace="actions"),
            beam_size=top_k,
            max_steps=max_steps,
            min_steps=min_steps,
            per_node_beam_size=int(max(top_k // 3, 1)),
        )
        return cls(
            vocab=vocab,
            decoder=decoder,
            action_embedder=action_embedder,
            observations_encoder=observations_encoder,
            beam_search=beam_search_,
            source_sampling_ratio=source_sampling_ratio,
            top_k=top_k,
            max_steps=max_steps,
            predict_mode=predict_mode,
            **kwargs,
        )


Model.register("sequence-model", "construct_decoder")(ProceduralPlanningModel)
