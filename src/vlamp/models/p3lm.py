# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import itertools
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from allennlp.common import Lazy
from allennlp.common.cached_transformers import AutoModel
from allennlp.common.params import Params, remove_keys_from_params
from allennlp.models.model import _DEFAULT_WEIGHTS
from allennlp.nn.beam_search import BeamSearch, Constraint, ConstraintStateType
from allennlp.training.metrics import Metric
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.models.basic_classifier import CategoricalAccuracy
from .planning_base_model import PlanningModel, SequenceCrossEntropyWithMask
from allennlp.nn.util import (
    batched_index_select,
    get_text_field_mask,
    get_token_ids_from_text_field_tensors,
    masked_mean,
    min_value_of_dtype,
    read_state_dict,
)
import json
from vlamp.flags import ObservationType, DEBUG
from transformers import AutoTokenizer, PreTrainedTokenizer
from vlamp.modules.observation_encoders.frame_encoders import (
    ObservationFeaturesEncoder,
)
from vlamp.modules.pretrained_transformer_decoder import (
    PretrainedTransfomerDecoder,
)
from vlamp.modules.pretrained_transformer_embedding import (
    PretrainedTransformerTokenEmbeddings,
)
import torch
import logging
from math import floor

LEVEL = logging.INFO
logger = logging.getLogger(__name__)


def _raise_value_error():
    """Used to raise error during list comprehension

    Raises:
        ValueError: _description_
    """
    raise ValueError


def get_max_length(mask: torch.Tensor) -> int:
    """Get the max length in batch using its mask

    Args:
        mask (torch.Tensor): binary mask with 0 for PAD

    Returns:
        int: max length in the batch
    """
    return int((torch.ones_like(mask) * mask).sum(dim=-1).max().item())


def remove_non_unique_sequences(
    mask: torch.BoolTensor, *inps: torch.Tensor
) -> List[torch.Tensor]:
    """Utility function that indexes into the batch dim using mask

    Args:
        mask (torch.BoolTensor): 1D binary tensor to index into the batch of inps. Should have length==inp.shape[0].

    Returns:
        _type_: _description_
    """
    res = []

    for inp in inps:
        assert len(mask.shape) == 1
        assert inp.shape[0] == mask.shape[0]
        res.append(inp[mask])

    return res


class PrefixLengthSpecificAccuracy(Metric):
    def __init__(self, intervals: List[Tuple[int, int]]) -> None:
        super().__init__()
        self.intervals = [(0, intervals[0][0])] + intervals + (intervals[-1][-1], 100)
        self.accuracies: Dict[Tuple[int, int], CategoricalAccuracy] = {
            interval: CategoricalAccuracy() for interval in self.intervals
        }

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        prefix_length: torch.LongTensor,  # (bs,)
    ) -> None:
        for (start, end), metric in self.accuracies.items():
            mask = (prefix_length < end) & (prefix_length >= start)
            metric(predictions=predictions, gold_labels=gold_labels, mask=mask)


KeyValueT = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


@Constraint.register("action-blocking")
class ActionBlocking(Constraint):
    def __init__(self, blocked: List[Union[str, int]], **kwagars) -> None:
        super().__init__(**kwagars)
        if isinstance(blocked[0], str):
            self.blocked: List[int] = [
                self.vocab.get_token_from_index(action, "actions") for action in blocked
            ]
        else:
            self.blocked = blocked

    def init_state(self, batch_size: int) -> ConstraintStateType:
        return [[{"dummy": 0}] for _ in range(batch_size)]

    def apply(
        self, state: ConstraintStateType, class_log_probabilities: torch.Tensor
    ) -> torch.Tensor:
        class_log_probabilities[:, :, self.blocked] = min_value_of_dtype(
            class_log_probabilities.dtype
        )
        return class_log_probabilities

    def _update_state(
        self, state: ConstraintStateType, last_prediction: torch.Tensor
    ) -> ConstraintStateType:
        beam_size = last_prediction.shape[-1]
        if len(state[0]) != beam_size:
            return [s * beam_size for s in state]
        else:
            return state


@Model.register("hfp3lm", "from_hf")
@Model.register("hfp3lm-for-eval")
class HFP3LM(PlanningModel):
    sk_attention_mask = "attention_mask"
    sk_last_hidden_state = "last_hidden_state"
    sk_key_value_prefix = "past_key_values"

    def __init__(
        self,
        vocab: Vocabulary,
        beam_search: BeamSearch,
        token_embedder: PretrainedTransformerTokenEmbeddings,  # HF transformer model name
        decoder: PretrainedTransfomerDecoder,
        tokenizer: PreTrainedTokenizer,
        observations_encoder: ObservationFeaturesEncoder,
        all_actions_token_indices: torch.Tensor,
        all_actions_token_indices_mask: torch.BoolTensor,
        new_tokens_loss_weight: Optional[float] = None,
        predict_mode: bool = False,
        observation_type: ObservationType = "pre",
        random_obs_masking_ratio: float = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            vocab, beam_search=beam_search, predict_mode=predict_mode, **kwargs
        )
        self.random_obs_masking_ratio = random_obs_masking_ratio
        self.token_embedder = token_embedder
        self.decoder = decoder
        self.observations_encoder = observations_encoder
        self.token_accuracy = CategoricalAccuracy()
        self.tokenizer = tokenizer
        self.observation_type = observation_type
        self.beam_search = beam_search
        self._num_obs_tokens = 0
        self._pad_action_idx = self.vocab.get_token_index(
            self.vocab._padding_token, "actions"
        )
        # override the base action loss if required
        if new_tokens_loss_weight is not None:
            class_weights = torch.ones(len(self.tokenizer))
            for tok, idx in self.tokenizer.get_added_vocab().items():
                class_weights[idx] = new_tokens_loss_weight
            self.action_loss = SequenceCrossEntropyWithMask(class_weight=class_weights)
        self.setup_indices_buffer(
            all_actions_token_indices,
            all_actions_token_indices_mask,
        )
        self.max_action_tokens_length = self.tokenized_input_ids.shape[-1]
        self._num_actions = self.tokenized_input_ids.shape[0]
        if self.use_task:
            self.task_embedding: torch.nn.Embedding = torch.nn.Embedding(
                vocab.get_vocab_size("tasks"), self.token_embedder.get_output_dim()
            )
            with torch.no_grad():
                self.task_embedding.weight.copy_(
                    self.token_embedder.get_weight().mean(dim=0).unsqueeze(0)
                )
        self.setup_logging()

    @property
    def num_obs_tokens(self) -> int:
        return self._num_obs_tokens

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def pad_action_idx(self) -> int:
        return self._pad_action_idx

    @property
    def end_action_idx(self) -> int:
        return self._end_action_idx

    @classmethod
    def from_hf(
        cls,
        vocab: Vocabulary,
        model_name: str,
        observations_encoder: ObservationFeaturesEncoder,
        beam_search: Lazy[BeamSearch] = Lazy(BeamSearch),
        new_tokens: Optional[Dict[str, str]] = None,
        random_lm_weights: bool = False,
        top_k: int = 10,
        per_node_beam_size: Optional[int] = None,
        max_steps: int = 4,
        min_steps: int = 1,
        action_only: bool = False,
        **kwargs,
    ):
        model = AutoModel.from_pretrained(model_name)

        if random_lm_weights:
            logger.info("Randomly initializing LM weights.")
            model.init_weights()
        decoder = PretrainedTransfomerDecoder(pretrained_model=model)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if new_tokens is not None:
            tokenizer.add_tokens(list(new_tokens.values()))
            end_action_text = "".join(
                [
                    new_tokens.get("end_action_text", ""),
                    new_tokens.get("end_of_action_text", ""),
                ]
            )
        else:
            end_action_text = ""
        end_action_idx: Optional[int] = None
        num_new_tokens = len(new_tokens) if new_tokens is not None else 0

        # Map the action vocabulary to their tokenized token indices array
        temp: List[str] = []

        for i in range(vocab.get_vocab_size("actions")):
            token = vocab.get_token_from_index(i, "actions")
            # remember the PAD action id
            if token == vocab._oov_token:
                if not tokenizer.unk_token:
                    tokenizer.add_special_tokens({"unk_token", "<|unk|>"})
                    num_new_tokens += 1
                temp.append(tokenizer.unk_token)
            elif token == vocab._padding_token:
                if not tokenizer.pad_token:
                    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                    num_new_tokens += 1
                temp.append(tokenizer.pad_token)
                pad_action_idx = i
            elif token == end_action_text:
                end_action_idx = i
                temp.append(token)
            else:
                temp.append(token)
        if end_action_idx is None:
            raise ValueError

        tokenized: Dict = tokenizer(temp, padding=True)
        # mask_for_actions_to_consider[pad_action_id] = False
        for k, v in tokenized.items():
            if isinstance(v, list):
                tokenized[k] = torch.tensor(v)
            if "attention" in k:
                tokenized[k][pad_action_idx] = 0

        token_embedder = PretrainedTransformerTokenEmbeddings.by_name(model_name)(
            model_name,
            pretrained_model=model,
            num_extensions=num_new_tokens,
        )

        assert observations_encoder.get_output_dim() == token_embedder.get_output_dim()
        beam_search_ = beam_search.construct(
            vocab=vocab,
            end_index=end_action_idx,
            beam_size=top_k,
            per_node_beam_size=per_node_beam_size or int(top_k // 3),
            max_steps=max_steps,
            min_steps=min_steps,
            constraints=[
                Lazy(
                    ActionBlocking,
                    constructor_extras=dict(
                        blocked=(tokenized["attention_mask"].sum(-1) == 0)
                        .nonzero()
                        .squeeze(-1)
                        .tolist()
                    ),
                )
            ],
        )
        instance = cls(
            vocab,
            beam_search_,
            token_embedder,
            decoder,
            tokenizer,
            observations_encoder,
            tokenized["input_ids"],
            tokenized["attention_mask"],
            top_k=top_k,
            action_only=action_only,
            max_steps=max_steps,
            **kwargs,
        )

        return instance

    @classmethod
    def _load(
        cls,
        config,  # : Params
        serialization_dir,  #: Union[str, PathLike],
        weights_file,  #: Optional[Union[str, PathLike]] = None,
        cuda_device: int = -1,
    ) -> "HFP3LM":
        # device_map = {
        #    0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        #    1: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        #    2: [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
        # }
        # device_map = {
        #    0: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        #    1: [9, 10, 11, 12, 13, 14, 15, 16, 17],
        #    2: [18, 19, 20, 21, 22, 23, 24, 25, 26],
        #    3: [27, 28, 29, 30, 31, 32, 33, 34, 35]
        # }
        """Based on the allennlp.models.model._load()"""
        weights_file = weights_file or os.path.join(serialization_dir, _DEFAULT_WEIGHTS)

        # Load vocabulary from file
        vocab_dir = os.path.join(serialization_dir, "vocabulary")
        # If the config specifies a vocabulary subclass, we need to use it.
        vocab_params = config.get("vocabulary", Params({}))
        vocab_choice = vocab_params.pop_choice(
            "type", Vocabulary.list_available(), True
        )
        vocab_class, _ = Vocabulary.resolve_class_name(vocab_choice)
        vocab = vocab_class.from_files(
            vocab_dir, vocab_params.get("padding_token"), vocab_params.get("oov_token")
        )

        model_params = config.get("model")
        device_map = model_params.get("device_map", None)
        logger.warning("HARD CODED device map")
        device_map = json.loads(
            '{"0": [0,1,2], "1": [3,4,5], "2": [6,7,8], "3": [9,10,11]}'
        )
        if device_map:
            # device_map = json.loads(device_map)
            device_map = {int(k): v for k, v in device_map.items()}

        # The experiment config tells us how to _train_ a model, including where to get pre-trained
        # embeddings/weights from. We're now _loading_ the model, so those weights will already be
        # stored in our model. We don't need any pretrained weight file or initializers anymore,
        # and we don't want the code to look for it, so we remove it from the parameters here.
        remove_keys_from_params(model_params)
        if "type" in model_params:
            del model_params["type"]

        model = Model.from_params(
            vocab=vocab,
            params=model_params,
            serialization_dir=serialization_dir,
            constructor_to_call=cls.from_hf,
            constructor_to_inspect=cls.from_hf,
        )
        model = cast(HFP3LM, model)
        # Load state dict. We pass `strict=False` so PyTorch doesn't raise a RuntimeError
        # if the state dict is missing keys because we handle this case below.
        model_state = read_state_dict(
            weights_file, cuda_device=cuda_device if device_map is None else -1
        )
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        if device_map:
            moved = False
            for dev, layers in device_map.items():
                if 0 in layers:
                    model.move_buffers(device=dev)
                    model.token_embedder.to(device=dev)
                    model.observations_encoder.to(device=dev)
                    if model.use_task:
                        model.task_embedding.to(device=dev)
                    moved = True
            assert moved
        else:
            if cuda_device >= 0:
                model.cuda(cuda_device)
            else:
                model.cpu()

        # Modules might define a class variable called `authorized_missing_keys`,
        # a list of regex patterns, that tells us to ignore missing keys that match
        # any of the patterns.
        # We sometimes need this in order to load older models with newer versions of AllenNLP.

        def filter_out_authorized_missing_keys(module, prefix=""):
            nonlocal missing_keys
            for pat in getattr(module.__class__, "authorized_missing_keys", None) or []:
                missing_keys = [
                    k
                    for k in missing_keys
                    if k.startswith(prefix) and re.search(pat[len(prefix) :], k) is None
                ]
            for name, child in module._modules.items():
                if child is not None:
                    filter_out_authorized_missing_keys(child, prefix + name + ".")

        filter_out_authorized_missing_keys(model)
        # remove task embedding from unexpected keys if not used but in the state dict
        if "task_embedding.weight" in unexpected_keys and not model_params.get(
            "task_embedding"
        ):
            unexpected_keys.remove("task_embedding.weight")
        if unexpected_keys or missing_keys:

            raise RuntimeError(
                f"Error loading state dict for {model.__class__.__name__}\n\t"
                f"Missing keys: {missing_keys}\n\t"
                f"Unexpected keys: {unexpected_keys}"
            )
        if device_map is not None:
            model.decoder.pretrained_model.parallelize(device_map)
        return model

    def setup_logging(self):
        if DEBUG:
            self.token_embedder.should_log_activations = True
            self.observations_encoder.should_log_activations = True

    def setup_indices_buffer(
        self,
        all_actions_token_indices: torch.Tensor,
        all_actions_token_indices_mask: torch.BoolTensor,
    ):
        self.register_buffer("tokenized_input_ids", all_actions_token_indices)
        self.register_buffer("tokenized_attention_mask", all_actions_token_indices_mask)
        # ISSUE: The following line is only to allow loading old models
        self.register_buffer(
            "tokenized_actions_to_consider", all_actions_token_indices_mask.sum(-1)
        )

    def move_buffers(self, device) -> None:
        self.tokenized_input_ids = self.tokenized_input_ids.to(device=device)
        self.tokenized_attention_mask = self.tokenized_attention_mask.to(device=device)
        self.tokenized_actions_to_consider = self.tokenized_actions_to_consider.to(
            device=device
        )

    @classmethod
    def move_pads_to_end(cls, index, *args: torch.Tensor) -> List[torch.Tensor]:
        res = [
            torch.gather(arg, -1, index)
            if arg.shape == index.shape
            else torch.gather(arg, -2, index.unsqueeze(-1).expand_as(arg))
            if arg.shape[:-1] == index.shape
            else _raise_value_error()
            for arg in args
        ]
        return res

    def get_last_hidden_state(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.BoolTensor,
        add_dim: bool = True,
    ) -> torch.Tensor:
        # last_non_pad_index = attention_mask.sum(-1) - 1
        last_non_pad_index = (
            torch.arange(
                attention_mask.shape[-1], 0, -1, device=attention_mask.device
            ).unsqueeze(0)
            + torch.arange(
                10000, 10000 + attention_mask.shape[-1], 1, device=attention_mask.device
            ).unsqueeze(0)
            * (~attention_mask)
        ).argmin(-1)
        last_hidden_state = batched_index_select(
            hidden_states, last_non_pad_index
        )  # (bs, num_actions, hidden_size)

        if add_dim:
            last_hidden_state = last_hidden_state.unsqueeze(
                -2
            )  # (bs*num_actions, 1, hidden_size)

        return last_hidden_state

    def init_decoder_state(
        self,
        prefix_tokens: torch.Tensor,
        prefix_attention_mask: torch.BoolTensor,
    ) -> Dict[str, torch.Tensor]:
        # Create decoder state by encoding the prefix
        prefix_position_ids = self.get_position_ids(prefix_attention_mask)
        first_step_outputs = self.decoder(
            prefix_tokens,
            attention_mask=prefix_attention_mask,
            position_ids=prefix_position_ids,
            use_cache=True,
        )
        past_key_values = first_step_outputs.past_key_values
        last_hidden_state = self.get_last_hidden_state(
            first_step_outputs.last_hidden_state.to(
                device=prefix_attention_mask.device
            ),
            prefix_attention_mask,
        )
        decoder_state = {
            self.sk_attention_mask: prefix_attention_mask,
            self.sk_last_hidden_state: last_hidden_state,
            **self.key_values_to_dict(past_key_values),
        }

        return decoder_state

    @torch.no_grad()
    def _get_nearest_token_indices(self, reps: torch.Tensor) -> torch.LongTensor:
        wt = self.token_embedder.get_weight()
        nearest_tokens = torch.nn.functional.linear(
            reps.to(device=wt.device), wt, bias=None
        ).max(-1)[1]
        return nearest_tokens

    def forward_next_token_prediction(
        self,
        observations: torch.Tensor,  # (batch, seq, num_obs_tokens, obs_emb_size)
        actions: TextFieldTensors,  # Dict (batch, seq, num_act_tokens)
        task: Optional[TextFieldTensors] = None,
        unique_sequence_mask: Optional[torch.BoolTensor] = None,
        meta: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        result = {}
        (
            tokens,
            action_token_ids,
            action_token_mask,
            observation_mask,
            attention_mask,
        ) = self.prepare_decoder_input(
            observations, actions, unique_sequence_mask, task
        )

        if self.random_obs_masking_ratio and self.training > 0:
            with torch.no_grad():
                b_mask, s_mask = torch.nonzero(
                    observation_mask * attention_mask, as_tuple=True
                )
                non_zero = len(b_mask)
                number_of_masks = int(floor(non_zero * self.random_obs_masking_ratio))
                if number_of_masks > 0:
                    index = torch.ones_like(b_mask, dtype=tokens.dtype).multinomial(
                        number_of_masks, replacement=False
                    )
                    attention_mask[b_mask[index], s_mask[index]] = False
                    observation_mask[b_mask[index], s_mask[index]] = False

        if tokens.shape[0] == 0:  # no unique sequences in batch
            assert not self.training

            return {}
        decoder_output: torch.Tensor = self.decoder(
            tokens[:, :-1, :],
            attention_mask=attention_mask[:, :-1],
            position_ids=None,  # let the decoder create increasing positions
        ).last_hidden_state  # (batch, seq*(num_act_tokens+num_obs_tokens)-1, emb_size)
        (
            action_loss,
            decoder_action_logits,
            decoder_action_mask,
            decoder_action_ids,
        ) = self.compute_action_next_token_loss(
            action_token_ids, action_token_mask, decoder_output
        )

        result.update(
            {
                "decoder_action_logits": decoder_action_logits,
                "decoder_action_mask": decoder_action_mask,
                "decoder_action_ids": decoder_action_ids,
                "decoder_observation_mask": observation_mask[:, 1:],
                "action_loss": action_loss,
            }
        )

        self.action_loss_metric(action_loss)

        if self.predict_mode:
            with torch.no_grad():
                nearest_tokens = torch.nn.functional.linear(
                    tokens[:, 1:, :], self.token_embedder.get_weight(), bias=None
                ).max(-1)[
                    1
                ]  # (bs, seq*(na+no)-1, tokens_vocab_size)
            result["nearest_tokens_to_decoder_input"] = nearest_tokens

        if self.training:
            if not self.action_only:
                obs_loss = self.observation_loss(
                    decoder_output,
                    tokens[:, 1:, :].contiguous(),
                    observation_mask[:, 1:].contiguous(),
                )
            else:
                obs_loss = 0
            weighted_obs_loss = self.observation_loss_weight * obs_loss
            self.observation_loss_metric(weighted_obs_loss)
            result.update(
                {
                    "observation_loss": obs_loss,
                    "loss": action_loss + weighted_obs_loss,
                }
            )

        self.token_accuracy(
            decoder_action_logits, decoder_action_ids, mask=decoder_action_mask
        )

        return result

    def compute_action_next_token_loss(
        self,
        action_token_ids: torch.LongTensor,
        action_token_mask: torch.BoolTensor,
        decoder_output: torch.Tensor,  # (bs, seq*(na+no), hidden_size)
    ):
        decoder_action_token_mask = action_token_mask[
            :, 1:
        ].contiguous()  # (bs, seq*(na+no)-1)
        assert (
            decoder_action_token_mask.dtype == torch.bool
        ), "Expect boolean mask for indexing"
        action_ids = action_token_ids[:, 1:].contiguous()  # (bs, seq*(na+no)-1)
        action_logits = torch.nn.functional.linear(
            decoder_output, self.token_embedder.get_weight(), bias=None
        )  # (bs, seq*(na+no), tokens_vocab_size)
        action_loss = self.action_loss(
            action_logits, action_ids, decoder_action_token_mask
        )

        return action_loss, action_logits, decoder_action_token_mask, action_ids

    def compute_action_loss_(
        self, action_token_ids, action_token_mask, decoder_output, flatten: bool = False
    ):
        decoder_output_emb_size = decoder_output.shape[-1]
        decoder_action_token_mask = action_token_mask[:, 1:].flatten()
        decoder_action_output = decoder_output.view(-1, decoder_output_emb_size)[
            decoder_action_token_mask
        ]  # (batch*num_actions_in_all_seq, emb_size)
        assert (
            decoder_action_token_mask.dtype == torch.bool
        ), "Expect boolean mask for indexing"
        action_ids = action_token_ids[:, 1:].flatten()[decoder_action_token_mask]
        action_logits = torch.nn.functional.linear(
            decoder_action_output, self.token_embedder.get_weight(), bias=None
        )  # (batch* total_num_actions, actions_vocab_size)
        action_loss = self.action_loss(action_logits, action_ids)

        return action_logits, action_loss

    def prepare_decoder_input(
        self,
        observations: torch.Tensor,
        actions: TextFieldTensors,
        unique_sequence_mask: Optional[torch.BoolTensor] = None,
        task: Optional[TextFieldTensors] = None,
    ):
        (
            _tokens,
            _action_token_ids,
            _action_token_mask,
            _observation_mask,
            _attention_mask,
        ) = self.create_joint_action_observation_sequence(observations, actions, task)
        # Note: variables with prefix _ are the ones before rearranging of padding
        # while the ones with no _ are after moving all the pads to the end

        # Remove repeated sequences if we are are evaluating and not predicting

        if unique_sequence_mask is not None and not self.predict_mode:
            (
                _tokens,
                _action_token_ids,
                _action_token_mask,
                _observation_mask,
                _attention_mask,
            ) = remove_non_unique_sequences(
                unique_sequence_mask,
                _tokens,
                _action_token_ids,
                _action_token_mask,
                _observation_mask,
                _attention_mask,
            )

        # Rearrange
        non_padded_positions = self.index_for_moving_pads_to_end(
            _tokens, _attention_mask
        )
        (
            tokens,
            attention_mask,
            action_token_mask,
            action_token_ids,
            observation_mask,
        ) = self.move_pads_to_end(
            non_padded_positions,
            _tokens,
            _attention_mask,
            _action_token_mask,
            _action_token_ids,
            _observation_mask,
        )

        return (
            tokens,
            action_token_ids,
            action_token_mask,
            observation_mask,
            attention_mask,
        )

    def index_for_moving_pads_to_end(self, tokens, attention_mask):
        # Rearrange tokens to move all PADs at the end.
        increasing_positions = torch.arange(
            tokens.shape[-2], device=tokens.device, dtype=torch.long
        ).expand_as(
            attention_mask
        )  # (1, seq*(na+no))
        increasing_positions = increasing_positions + 1000 * (
            ~attention_mask
        )  # High value where there are pads
        max_position = increasing_positions.shape[-1] - 1
        non_padded_positions = increasing_positions.sort(dim=-1, stable=True)[1]
        # (batch, seq*(na+no))
        # Clamp the high values to max
        non_padded_positions = non_padded_positions.clamp_max_(max_position)

        return non_padded_positions

    def create_joint_action_observation_sequence(
        self,
        observations: torch.Tensor,
        actions: TextFieldTensors,
        task: Optional[TextFieldTensors] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: To make the model truly flexible, we need to accommodate different lengths of actions and observations
        # Patters we could get with action length (in num tokens) 4 and obs length 3:
        # 1. Post: aaaa-ooo aaaa-ooo ...
        # 2. Pre:  ooo-aaaa ooo-aaaa ...
        # 3. Both: ooo-aaaa-ooo ooo-aaaa-oooo ...
        # get action token embeddings
        # ISSUE: We are not using the unique_sequence_mask at this time. This will make the metrics incorrect at inference time
        (
            action_token_ids_,
            action_token_mask_,
        ) = self.convert_actions_into_tokenized_input(actions)
        # (batch, seq, num_act_tokens) with False for PAD
        bs, seq, num_act_toks = action_token_ids_.shape
        action_token_embeddings_ = self.token_embedder(
            action_token_ids_
        )  # (batch, seq, num_act_tokens, token_emb_size)
        observations_embeddings_ = self.observations_encoder(
            observations
        )  # (batch, seq, num_obs_tokens, token_emb_size)
        bs_, seq_, num_obs_toks, token_emb_size = observations_embeddings_.shape

        # if self.predict_mode:
        #    self.add_to_prediction_buffer(
        #        "observation_input_reps",
        #        observations_embeddings_.view(bs_, seq_, num_obs_toks, token_emb_size),
        #    )

        assert bs_ == bs
        assert seq_ == seq
        if not self.action_only:
            obs_mask_ = torch.ones(
                seq,
                num_obs_toks,
                dtype=torch.bool,
                device=observations_embeddings_.device,
            )  # (seq, num_obs_tokens)
        else:
            obs_mask_ = torch.zeros(
                seq,
                num_obs_toks,
                dtype=torch.bool,
                device=observations_embeddings_.device,
            )  # (seq, num_obs_tokens)
        task_mask = torch.ones(
            bs_, 1, dtype=torch.bool, device=observations_embeddings_.device
        )
        # Do we need to make this causal or the model does it? Answer: At least for GPT-2 from HF, the model does it.
        obs_mask_ = obs_mask_.expand(bs, seq, num_obs_toks)
        # observation_mask: observation tokens after last action are considered as obs pads. do not predict after last action
        observation_mask_ = (action_token_mask_.sum(dim=-1) > 0).unsqueeze(
            -1
        ) * obs_mask_  # shape (batch, seq, num_obs)
        # attention_mask: Has True for all non pad action tokens and non pad observation tokens.

        if self.observation_type in ["post", "pre"]:
            # ISSUE: Currently the else block will never be executed
            # and it is on the dataset reader to provide observations that can be put after the action.
            attention_mask = torch.cat(
                [action_token_mask_, observation_mask_], dim=-1
            ).view(
                bs, -1
            )  # (batch, seq*(num_act_tokens + num_obs_tokens))
            if self.use_task:
                attention_mask = torch.cat(
                    [task_mask, attention_mask], dim=-1
                )  # (batch, 1+ seq(na+no))
            if not self.action_only:
                action_token_mask = torch.cat(
                    [action_token_mask_, ~obs_mask_], dim=-1
                ).view(
                    bs, -1
                )  # (batch, seq*(na+no)) where True is only for action tokens excluding action pads

            else:
                action_token_mask = torch.cat(
                    [action_token_mask_, obs_mask_], dim=-1
                ).view(
                    bs, -1
                )  # (batch, seq*(na+no)) where True is only for action tokens excluding action pads
            if self.use_task:
                action_token_mask = torch.cat([~task_mask, action_token_mask], dim=-1)
            observation_mask = torch.cat(
                [
                    torch.zeros_like(action_token_mask_, dtype=torch.bool),
                    observation_mask_,
                ],
                dim=-1,
            ).view(
                bs, -1
            )  # (batch, seq*(na+no)) where True is only for observation tokens excluding observation pads
            if self.use_task:
                observation_mask = torch.cat([~task_mask, observation_mask], dim=-1)
            action_token_ids = torch.cat(
                [action_token_ids_, torch.zeros_like(obs_mask_, dtype=torch.long)],
                dim=-1,
            ).view(
                bs, -1
            )  # (batch, seq*(na+no))
            if self.use_task:
                action_token_ids = torch.cat(
                    [torch.zeros_like(task_mask, dtype=torch.long), action_token_ids],
                    dim=-1,
                )

            tokens = torch.cat(
                [action_token_embeddings_, observations_embeddings_], dim=-2
            ).view(bs, -1, token_emb_size)
            if self.use_task:
                task_ = get_token_ids_from_text_field_tensors(task)
                task_embeddings = self.task_embedding(task_)  # (bs, 1,emb_size)
                tokens = torch.cat([task_embeddings, tokens], dim=-2)
        elif self.observation_type == "pre":
            attention_mask = torch.cat(
                [observation_mask_, action_token_mask_], dim=-1
            ).view(
                bs, -1
            )  # (batch, seq*(num_act_tokens + num_obs_tokens))
            action_token_mask = torch.cat(
                [~obs_mask_, action_token_mask_], dim=-1
            ).view(
                bs, -1
            )  # (batch, seq*(na+no)) where True is only for action tokens excluding action pads
            observation_mask = torch.cat(
                [
                    observation_mask_,
                    torch.zeros_like(action_token_mask_, dtype=torch.bool),
                ],
                dim=-1,
            ).view(
                bs, -1
            )  # (batch, seq*(na+no)) where True is only for observation tokens excluding observation pads
            action_token_ids = torch.cat(
                [
                    torch.zeros_like(obs_mask_, dtype=torch.long),
                    action_token_ids_,
                ],
                dim=-1,
            ).view(
                bs, -1
            )  # (batch, seq*(na+no))

            tokens = torch.cat(
                [
                    observations_embeddings_,
                    action_token_embeddings_,
                ],
                dim=-2,
            ).view(bs, -1, token_emb_size)
        else:
            raise ValueError

        # (batch, seq*(num_act_tokens+num_obs_tokens), emb_size)

        return (
            tokens,
            action_token_ids,
            action_token_mask,
            observation_mask,
            attention_mask.bool(),
        )

    def get_extra_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "action_token_accuracy": self.token_accuracy.get_metric(reset),
        }

    def convert_actions_into_tokenized_input(
        self, actions: TextFieldTensors
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        action_ids = get_token_ids_from_text_field_tensors(
            actions
        )  # (batch, num_steps)
        action_seq_mask = get_text_field_mask(actions)  # (batch, num_steps)

        return self._convert_actions_into_tokenized_input(action_ids, action_seq_mask)

    def _convert_actions_into_tokenized_input(
        self, action_ids: torch.Tensor, action_seq_mask: torch.Tensor, trim: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, num_steps = action_ids.shape
        f_ = action_ids.flatten()
        action_ids = self.tokenized_input_ids[f_].reshape(
            bs, num_steps, -1
        )  # (bs, num_steps, tokens_per_step)
        action_mask = self.tokenized_attention_mask[f_].to(dtype=torch.bool).reshape(
            bs, num_steps, -1
        ) * action_seq_mask.unsqueeze(
            -1
        )  # (bs, num_steps, tokens_per_step)

        if trim:
            max_len = get_max_length(action_mask.view(-1, action_mask.shape[-1]))
            action_ids = action_ids[..., :max_len]
            action_mask = action_mask[..., :max_len]

        return action_ids, action_mask

    def repeat_interleave_batch(self, *inps: torch.Tensor) -> List[torch.Tensor]:
        if len(inps) == 1:
            return inps[0].repeat_interleave(self.num_actions, dim=0)
        else:
            return [inp.repeat_interleave(self.num_actions, dim=0) for inp in inps]

    def repeat_uninterleave_batch(self, *inps: torch.Tensor) -> List[torch.Tensor]:
        # inp: (bs*num_actions, ...)
        # out: (bs, num_actions, ...)
        if len(inps) == 1:
            return inps[0].reshape(-1, self.num_actions, *inps[0].shape[1:])
        else:
            return [inp.reshape(-1, self.num_actions, *inp.shape[1:]) for inp in inps]

    def _take_decoding_step(
        self,
        tokens: torch.Tensor,
        past_key_values: Tuple,
        attention_mask: torch.BoolTensor,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        outputs = self.decoder(
            tokens,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )

        return outputs.last_hidden_state, outputs.past_key_values

    def set_num_obs_tokens(self, observations: torch.Tensor) -> None:
        self._num_obs_tokens = observations.shape[-2]

    def forward_beam_search(
        self,
        prefix_observations,
        prefix_actions,
        task: Optional[TextFieldTensors] = None,
    ):
        (
            tokens,  # (bs, seq*(na+no), emb_size)
            action_token_ids,
            action_token_mask,
            observation_mask,
            attention_mask,
        ) = self.prepare_decoder_input(prefix_observations, prefix_actions, task=task)

        # Take first decoding step
        # pos ids for the prefix
        # Make num_actions copies of the prefix in the batch dimension
        # with first num_action entries corresponding to first instance and so on.
        # (
        #    tokens,  # (bs*num_actions, seq, emb_size) each element in dim 0 is consecutively repeated self.num_actions times.
        #    position_ids,
        #    attention_mask,
        # ) = self.repeat_interleave_batch(tokens, position_ids, attention_mask)
        # Create decoder state by encoding the prefix
        decoder_state = self.init_decoder_state(tokens, attention_mask)

        all_top_k_predictions, log_probabilities = self.beam_search.search(
            torch.zeros(tokens.shape[0], dtype=torch.long),
            decoder_state,
            self.take_search_step,
        )
        # if self.predict_mode:  # reshape observations_prediction_reps
        #    if "observation_prediction_reps" in self.prediction_buffer:
        #        bs, beam_size, _ = all_top_k_predictions.shape
        #        _ = self.prediction_buffer["observation_prediction_reps"].shape[1:]
        #        self.prediction_buffer["observation_prediction_reps"].reshape(
        #            bs, beam_size, *_
        #        )

        return all_top_k_predictions, log_probabilities

    def get_position_ids(self, attention_mask) -> torch.LongTensor:
        return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

    def reduce_state(
        self, state: Dict[str, torch.Tensor], action_ids: torch.LongTensor
    ) -> Dict[str, torch.Tensor]:
        action_ids = cast(torch.LongTensor, action_ids.unsqueeze(-1))  # (group, 1)
        reduced_state: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            trailing_dims = v.shape[2:]
            action_ids = action_ids.to(device=v.device)
            reduced_state[k] = v.gather(
                1,
                action_ids.reshape(-1, 1, *([1] * len(trailing_dims))).repeat(
                    1, 1, *trailing_dims
                ),
            ).squeeze(1)
        return reduced_state

    def take_search_step(
        self,
        last_predictions: torch.LongTensor,  # (group, num_actions)
        state: Dict[str, torch.Tensor],
        step: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if step == 0:
            temp_state = state
        else:
            temp_state = self.reduce_state(
                state, last_predictions
            )  # Index into the action repeats
        attention_mask = cast(
            torch.BoolTensor, temp_state["attention_mask"]
        )  # (group, seq)
        last_hidden_state = temp_state["last_hidden_state"]  # (group, 1, hidden_size)
        # if self.predict_mode and not self.action_only and step > 0:
        #    if "observation_prediction_reps" not in self.prediction_buffer:
        #        self.add_to_prediction_buffer(
        #            "observation_prediction_reps",
        #            temp_state["observation_prediction_reps"].unsqueeze(1),
        #        )
        #    else:
        #        self.prediction_buffer["observation_prediction_reps"] = torch.cat(
        #            [
        #                self.prediction_buffer["observation_prediction_reps"],
        #                temp_state["observation_prediction_reps"]
        #                .detach()
        #                .cpu()
        #                .unsqueeze(1),
        #            ],
        #            dim=-3,
        #        )

        past_key_values = self.get_key_values(
            temp_state
        )  # each Key or Value is (group, heads, seq, emb)
        (
            action_logits,
            attention_mask,  # (group, num_actions, seq)
            position_ids,  # expanded to (group, num_actions, seq)
            past_key_values,  # expanded to (group, num_actions, heads, seq, emb)
            last_hidden_state,  # expanded to (group, num_actions, 1, hidden_size)
            observation_predictions,  # None or (group, num_actions, num_obs_tokens, hidden_size)
        ) = self.next_action_logits(attention_mask, past_key_values, last_hidden_state)
        new_state = {
            "attention_mask": attention_mask,
            "last_hidden_state": last_hidden_state.to(device=attention_mask.device),
            **self.key_values_to_dict(past_key_values),
        }
        # if self.predict_mode and not self.action_only:
        #    new_state["observation_prediction_reps"] = observation_predictions
        return action_logits, new_state

    @classmethod
    def key_values_to_dict(
        cls, key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    ) -> Dict[str, torch.Tensor]:
        return {
            (cls.sk_key_value_prefix + f"_{k_or_v_str}_{i}"): k_or_v_tensor
            for i, k_v in enumerate(key_values)
            for k_or_v_tensor, k_or_v_str in zip(k_v, ["k", "v"])
        }

    def get_key_values(
        self, key_values_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        # ISSUE: Hard coding for GPT2.
        return tuple(
            (
                key_values_dict[self.sk_key_value_prefix + f"_k_{layer_num}"],
                key_values_dict[self.sk_key_value_prefix + f"_v_{layer_num}"],
            )
            for layer_num in range(self.decoder.num_attention_layers())
        )

    def _index_into_key_values(
        self,
        key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        start: int,
        end: Optional[int] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple(tuple(val[start:end] for val in tup) for tup in key_values)

    def next_action_logits(
        self,
        attention_mask: torch.BoolTensor,
        past_key_values: KeyValueT,
        last_hidden_state: torch.Tensor,
    ):
        bs, seq = attention_mask.shape
        attention_mask, last_hidden_state = self.repeat_interleave_batch(
            attention_mask, last_hidden_state
        )
        past_key_values = tuple(
            self.repeat_interleave_batch(k, v) for k, v in past_key_values
        )
        position_ids = self.get_position_ids(attention_mask)

        (
            _tokens,
            attention_mask,  # full mask including history
            position_ids,  # only for next action step
            _attention_mask,  # only for next action step
        ) = self.create_input_for_action_step(attention_mask, position_ids)

        outputs, past_key_values = self._take_decoding_step(
            _tokens,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        assert _attention_mask.shape[-1] == outputs.shape[-2]
        outputs = outputs.to(device=last_hidden_state.device)
        next_action_prediction_hidden = torch.cat(
            [last_hidden_state, outputs[:, :-1]],
            dim=-2,
        )  # (group*num_actions, max_action_length, hidden_size)
        last_hidden_state = self.get_last_hidden_state(
            outputs, attention_mask=_attention_mask
        )
        if self.predict_mode:
            observation_token_predictions_ = [last_hidden_state.clone()]
        if not self.action_only:
            for obs_tok_num in range(self.num_obs_tokens):
                _attention_mask = torch.ones(
                    attention_mask.shape[0],
                    1,
                    device=attention_mask.device,
                    dtype=torch.bool,
                )
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        _attention_mask,
                    ],
                    dim=-1,
                )
                position_ids = (
                    (attention_mask.long().cumsum(-1) - 1)
                    .clamp(min=0)[:, -1]
                    .unsqueeze(-1)
                )  # (bs, 1)

                out, past_key_values = self._take_decoding_step(
                    last_hidden_state,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                out = out.to(device=last_hidden_state.device)

                last_hidden_state = out[:, -1, :].unsqueeze(-2)
                if self.predict_mode:
                    _ = last_hidden_state.shape[1:]
                    if obs_tok_num < self.num_obs_tokens - 1:
                        observation_token_predictions_.append(last_hidden_state)
            if self.predict_mode:
                observation_token_predictions = self.repeat_uninterleave_batch(
                    torch.cat(observation_token_predictions_, dim=-2)
                )

        # predictions, metrics and loss for the step
        logits = torch.nn.functional.linear(
            next_action_prediction_hidden,
            self.token_embedder.get_weight(),
            bias=None,
        )  # (bs*num_actions, max_action_length, token_vocab_size)
        log_probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
        temp = log_probabilities.view(
            bs, self.num_actions, self.max_action_tokens_length, -1
        )  # (bs, num_actions, max_action_length, token_vocab_size)
        action_token_score = torch.gather(
            temp,
            -1,
            self.action_vocab_ids.unsqueeze(-1).repeat(bs, 1, 1, 1),
        ).squeeze(
            -1
        )  # (bs, num_actions, max_action_length)
        action_score = masked_mean(
            action_token_score,
            mask=self.action_attention_mask,
            dim=-1,
        )
        # Push the num_action repetitions to the 1st dim.
        (
            position_ids,
            attention_mask,
            last_hidden_state,
        ) = self.repeat_uninterleave_batch(
            position_ids, attention_mask, last_hidden_state
        )
        past_key_values = tuple(
            self.repeat_uninterleave_batch(k, v) for k, v in past_key_values
        )
        return (
            action_score,
            attention_mask,
            position_ids,
            past_key_values,
            last_hidden_state,
            observation_token_predictions
            if self.predict_mode and not self.action_only
            else None,
        )

    def create_input_for_action_step(
        self,
        past_attention_mask: torch.BoolTensor,
        past_position_ids: torch.LongTensor,
    ):
        bs = past_attention_mask.shape[0] // self.num_actions
        _attention_mask = self.action_attention_mask.repeat(
            bs,
            1,
        )
        position_ids = past_position_ids[:, -1].unsqueeze(
            -1
        ) + _attention_mask.long().cumsum(-1)
        attention_mask = torch.cat([past_attention_mask, _attention_mask], dim=-1)
        tokens = self.action_vocab_reps.repeat(bs, 1, 1)

        return tokens, attention_mask, position_ids, _attention_mask

    @property
    def action_vocab_reps(self) -> torch.Tensor:
        return self.token_embedder(self.tokenized_input_ids)

    @property
    def action_vocab_ids(self) -> torch.LongTensor:
        return self.tokenized_input_ids

    @property
    def action_attention_mask(self) -> torch.Tensor:
        return (self.tokenized_attention_mask).bool()

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
        pass
        # self.add_to_prediction_buffer(
        #    "observation_target_reps", self.observations_encoder(target_observations)
        # )

    def create_actions_vocab_reps(
        self,
        past_attention_mask: torch.BoolTensor,
        past_position_ids: torch.LongTensor,
    ):
        bs = past_attention_mask.shape[0] // self.num_actions
        _tokens = self.token_embedder(self.tokenized_input_ids).repeat(
            bs, 1, 1
        )  # (bs*num_actions, max_action_length, hidden_size)
        _attention_mask = self.tokenized_attention_mask.repeat(
            bs, 1
        ).bool()  # (bs, max_action_length)
        # take last position id and add the cumsum of new mask
        _position_ids = past_position_ids[:, -1].unsqueeze(
            -1
        ) + _attention_mask.long().cumsum(-1)
        # attention mask always has to contain the past
        _attention_mask = torch.cat(
            [past_attention_mask, _attention_mask], dim=-1
        ).bool()

        return _tokens, _attention_mask, _position_ids

    def make_observations_human_readable(self, output_dict) -> Dict:
        out = {}
        token_ids = self._get_nearest_token_indices(
            output_dict["observation_prediction_reps"]
        )
        bs, seq, num_tok = token_ids.shape

        out["predicted_observation_tokens"] = [
            self.tokenizer.convert_ids_to_tokens(batch.tolist())
            for batch in token_ids.view(bs, seq * num_tok)
        ]
        token_ids = self._get_nearest_token_indices(
            output_dict["observation_input_reps"]
        )
        bs, seq, num_tok = token_ids.shape
        out["input_observation_tokens"] = [
            self.tokenizer.convert_ids_to_tokens(batch.tolist())
            for batch in token_ids.view(bs, seq * num_tok)
        ]
        return out

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[torch.Tensor, List]]:
        assert self.predict_mode, "override model.predict_mode to True"
        output: Dict[str, Union[torch.Tensor, List]] = {}
        # output.update(output_dict)
        raw_out = self.preserve_raw_outputs(output_dict)
        actions_out = self.make_actions_human_readable(output_dict)
        return {**actions_out, **raw_out}
