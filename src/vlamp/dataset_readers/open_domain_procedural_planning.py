# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from itertools import chain
from typing import Any, Dict, Iterator, List, Optional, Union
from allennlp.common import Lazy
from allennlp.data import DatasetReader, Field, Instance, TokenIndexer, Tokenizer, Token
from allennlp.data.fields import ArrayField, ListField, TextField
from allennlp.data.tokenizers import PretrainedTransformerTokenizer, SpacyTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
import numpy as np
from vlamp.dataset_readers.common import FileReader, Step
from .procedural_planning import ClosedDomainProceduralPlanning, PrefixT, Transform
from vlamp.flags import DEBUG


class TokenizerExtenderMixin:
    def add_tokens(self, tokens: List[str]) -> None:
        raise NotImplementedError


class ExtendableTokenizer(Tokenizer, TokenizerExtenderMixin):
    default_implementation: Optional[str] = "pretrained_transformer_extendable"
    pass


@ExtendableTokenizer.register("pretrained_transformer_extendable")
class PretrainedTransformerExtendableTokenizer(
    PretrainedTransformerTokenizer, ExtendableTokenizer
):
    def add_tokens(self, tokens: List[str]) -> None:
        self.tokenizer.add_tokens(tokens)


@TokenIndexer.register(
    "pretrained_transformer_extendable",
)
class PretrainedTransformerExtendableIndexer(PretrainedTransformerIndexer):
    def __init__(
        self,
        model_name: str,
        namespace: str = "tags",
        max_length: int = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer: Optional[PretrainedTransformerExtendableTokenizer] = None,
        **kwargs,
    ) -> None:
        super(PretrainedTransformerIndexer, self).__init__(**kwargs)
        self._namespace = namespace
        if tokenizer is None:
            self._allennlp_tokenizer = PretrainedTransformerTokenizer(
                model_name, tokenizer_kwargs=tokenizer_kwargs
            )
        else:
            self._allennlp_tokenizer = tokenizer
        self._tokenizer = self._allennlp_tokenizer.tokenizer
        self._added_to_vocabulary = False

        self._num_added_start_tokens = len(
            self._allennlp_tokenizer.single_sequence_start_tokens
        )
        self._num_added_end_tokens = len(
            self._allennlp_tokenizer.single_sequence_end_tokens
        )

        self._max_length = max_length
        if self._max_length is not None:
            num_added_tokens = len(self._allennlp_tokenizer.tokenize("a")) - 1
            self._effective_max_length = (  # we need to take into account special tokens
                self._max_length - num_added_tokens
            )
            if self._effective_max_length <= 0:
                raise ValueError(
                    "max_length needs to be greater than the number of special tokens inserted."
                )


@DatasetReader.register("open-domain-procedural-planning", "for_pretrained_transformer")
class OpenDomainProceduralPlanning(ClosedDomainProceduralPlanning):
    def __init__(
        self,
        action_tokenizer: ExtendableTokenizer,
        empty_action_text: str = "<|noact|>",
        end_of_action_text: str = " <|eoa|>",
        **kwargs,
    ) -> None:
        super().__init__(
            empty_action_text=empty_action_text,
            end_of_action_text=end_of_action_text,
            **kwargs,
        )
        self.action_tokenizer = action_tokenizer
        self.end_of_action_text = end_of_action_text

    @classmethod
    def for_pretrained_transformer(
        cls,
        action_tokenizer: ExtendableTokenizer,
        action_token_indexer: Lazy[PretrainedTransformerExtendableIndexer],
        empty_action_text: str = "<|noact|>",
        end_of_action_text: str = " <|eoa|>",
        **kwargs,
    ) -> "OpenDomainProceduralPlanning":
        action_tokenizer.add_tokens([empty_action_text, end_of_action_text])
        action_token_indexers = {
            "actions": action_token_indexer.construct(tokenizer=action_tokenizer)
        }
        return cls(
            action_tokenizer,
            empty_action_text=empty_action_text,
            end_of_action_text=end_of_action_text,
            action_token_indexers=action_token_indexers,
            **kwargs,
        )

    def prepare_actions_field(
        self, steps: List[Dict[str, Union[float, str, None]]], candidates: bool = False
    ) -> Field:
        if candidates:
            steps = steps[
                0:1
            ]  # TODO: To make this general and not one step only. We need to create a new field like the FlagField.
            if self.all_actions is None:
                self.all_actions = [
                    step_desc.strip()
                    for task in self.file_reader.tasks
                    for _, step_desc in task.stepid2desc.items()
                ]
            return ListField(
                [
                    ListField(
                        [
                            TextField(self.action_tokenizer.tokenize(action_at_time))
                            for action_at_time in chain(
                                action["desc"] + self.end_of_action_text
                                if self.end_of_action_text
                                else action["desc"],
                                (
                                    a + self.end_of_action_text
                                    if self.end_of_action_text
                                    else a
                                    for a in self.all_actions
                                    if a != action["desc"]
                                ),
                            )
                        ]
                    )
                    for action in steps
                ]
            )  # (seq, candidates, num_tokens) is ListField, ListField, TextField
        else:
            return ListField(
                [
                    TextField(
                        (
                            self.action_tokenizer.tokenize(
                                action["desc"] + self.end_of_action_text
                                if self.end_of_action_text
                                and isinstance(action["desc"], str)
                                else action["desc"]
                            )
                        )
                    )  # type: ignore
                    for action in steps
                ]
            )

    def apply_token_indexers(self, instance: Instance) -> None:
        for action in instance["actions"].field_list:
            action.token_indexers = self.action_token_indexers
        if self.create_task_field:
            instance["task"].token_indexers = self.task_token_indexers  # type: ignore
        if self.create_prefixes:
            for action in instance["prefix_actions"].field_list:
                action.token_indexers = self.action_token_indexers  # type: ignore
            for action_or_candidates in instance["target_actions"].field_list:
                if isinstance(action_or_candidates, ListField):
                    for action in action_or_candidates.field_list:
                        action.token_indexers = self.action_token_indexers  # type: ignore
                else:
                    action_or_candidates.token_indexers = self.action_token_indexers
