# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from math import floor
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    TypeVar,
    TypedDict,
    Union,
    cast,
)
from allennlp.common import Registrable
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    ArrayField,
    Field,
    LabelField,
    MetadataField,
    TensorField,
)
from allennlp.data.instance import Instance
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPVisionModel
from .common import ActionNoiseT, FileReader, Step
from allennlp.common.lazy import Lazy
from allennlp.data import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import TextField
from dataclasses import asdict
from vlamp.flags import DEBUG, ObservationType

torch.set_num_threads(1)

import logging
import random

logger = logging.getLogger(__name__)


def _raise_value_error():
    raise ValueError


class InstanceFields(TypedDict):
    """Contents which form an instance"""

    x: ArrayField
    y: LabelField  #: types


T_ = TypeVar("T_")


PrefixT = Literal[None, "sample", "all"]

StepsT = List[Dict[str, Union[float, str, None, np.ndarray]]]


# ISSUE: Fix hardcoding of image frame transform


class Transform(Registrable):
    def __call__(self, inp: T_) -> T_:
        raise NotImplementedError


@Transform.register("pass-through")
class PassThrough(Transform):
    def __call__(self, inp: T_) -> T_:
        return inp


@Transform.register("clip")
class CLIPTransform(Transform):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        super().__init__()
        self.model_name = model_name
        self.transform = CLIPProcessor.from_pretrained(model_name)

    def __call__(self, images: T_) -> T_:
        res = self.transform(images=images)
        return res["pixel_values"][0]


@Transform.register("clip-embedder")
class CLIPEmbedderTransform(Transform):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        super().__init__()
        self.model_name = model_name
        self.preprocessor = CLIPProcessor.from_pretrained(model_name)
        self.embedder = CLIPVisionModel.from_pretrained(model_name)
        self.embedder.requires_grad_(False)

    def __call__(self, images: T_) -> T_:
        res = self.preprocessor(images=images, return_tensors="pt")
        preprocessed_image = res["pixel_values"]
        assert len(preprocessed_image.shape) == 4  # (1,C, H, W)
        assert preprocessed_image.shape[0] == 1
        # There could be pads in the seq dimension
        result = self.embedder(
            preprocessed_image
        ).last_hidden_state  # (1, num_clip_tokens, clip_vision_hidden_size)
        assert result.shape[0] == 1
        return result[0]


class ObservationFeaturesField(TensorField):
    def __init__(
        self,
        tensor: Union[torch.Tensor, np.ndarray],
        padding_value: Any = 0,
        dtype: Optional[Union[np.dtype, torch.dtype]] = None,
        min_sequence_padding_length: Optional[int] = None,
    ) -> None:
        super().__init__(tensor, padding_value, dtype)
        self.min_sequence_padding_length = min_sequence_padding_length

    def get_padding_lengths(self) -> Dict[str, int]:
        padding_lengths = super().get_padding_lengths()
        if self.min_sequence_padding_length is not None:
            padding_lengths["dimension_0"] = max(
                padding_lengths["dimension_0"], self.min_sequence_padding_length
            )
        return padding_lengths


ActionNoiseLocationT = Literal["random", "end"]


ActionNoiseReplacementProbT = float


@DatasetReader.register("closed-domain-procedural-planning")
class ClosedDomainProceduralPlanning(DatasetReader):
    def __init__(
        self,
        file_reader: FileReader,
        image_features_transform: Transform = PassThrough(),
        image_frames_transform: Transform = PassThrough(),
        empty_action_text: str = "",  # "<|noact|>"
        end_of_action_text: str = "",  # use  for transformer models " <|eoa|>",
        start_action_text: str = "",  # <|sact|> for transformers
        end_action_text: str = "",  # <|eact|> for tr
        create_prefixes: PrefixT = None,
        debug: bool = DEBUG,
        create_task_field: bool = False,
        move_last_prefix_action_to_begining: bool = False,
        minimum_prefix_length: int = 1,
        minimum_target_length: int = 4,
        observation_type: ObservationType = "post",
        action_noise_type: ActionNoiseT = "none",
        action_noise_location: ActionNoiseLocationT = "random",
        action_noise_replacement_prob: ActionNoiseReplacementProbT = 0.5,
        no_history: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )
        self.action_noise_type = action_noise_type
        self.action_noise_location = action_noise_location
        self.action_noise_replacement_prob = action_noise_replacement_prob
        self.observation_type = observation_type
        self.minimum_target_length = minimum_target_length
        self.end_of_action_text = end_of_action_text
        self.create_task_field = create_task_field
        self.file_reader: FileReader = file_reader
        self.action_token_indexers = {
            "actions": SingleIdTokenIndexer(
                namespace="actions",
            )
        }  # ISSUE: If adding an end_token. Make sure that you repeat the last observation.
        #           Otherwise the length of action and observation sequences won't match.
        #           Currently there does not seem to be an easy way to enforce this.
        #           We could add the tokens manually instead of relying on the indexer,
        #           but we need to make sure to do it during predictions as well.
        self.target_action_token_indexers = {
            "actions": SingleIdTokenIndexer(
                namespace="actions",
                token_min_padding_length=self.minimum_target_length,
            )
        }
        self.task_token_indexers = (
            ({"task": SingleIdTokenIndexer(namespace="tasks")})
            if create_task_field
            else None
        )
        self.image_features_transform = image_features_transform
        self.image_frames_transform = image_frames_transform
        self.empty_action_text = empty_action_text
        self.start_action_text = start_action_text
        self.end_action_text = end_action_text
        self.create_prefixes = create_prefixes
        self.debug = debug
        self.all_actions: List[str] = None
        self.move_last_prefix_action_to_begining = move_last_prefix_action_to_begining
        self.minimum_prefix_length = minimum_prefix_length
        self.no_history = no_history

    def text_to_instances(  # type:ignore
        self,
        steps: StepsT,
        task_idx: Optional[str],
        task: Optional[Dict[str, Any]],
        **kwargs,
    ) -> Iterator[Instance]:
        steps = self.add_steps(steps)
        num_steps = len(steps)
        if self.create_prefixes is None:
            yield self.text_to_instance(
                steps,
                task_idx,
                task,
                prefix_start=0,
                prefix_end=None,
                unique_sequence=True,
                **kwargs,
            )
        else:
            assert (
                num_steps - self.minimum_prefix_length >= 1
            )  # at least one step to predict
            if self.create_prefixes == "sample":
                if num_steps - self.minimum_prefix_length == 1:
                    prefix_end = self.minimum_prefix_length
                else:
                    prefix_end = self.minimum_prefix_length + np.random.randint(
                        0, num_steps - self.minimum_prefix_length
                    )
                prefix_end_range = range(prefix_end, prefix_end + 1)
            elif self.create_prefixes == "all":
                prefix_end_range = range(
                    self.minimum_prefix_length,
                    max(
                        self.minimum_prefix_length + 1,
                        num_steps,
                    ),
                )
            else:
                raise ValueError(
                    f"Invalid value ({self.create_prefix}) for create_prefix"
                )
            for i, prefix_end_ in enumerate(prefix_end_range):
                yield self.text_to_instance(
                    steps,
                    task_idx,
                    task,
                    prefix_start=0 if not self.no_history else prefix_end_ - 1,
                    prefix_end=prefix_end_,
                    unique_sequence=(
                        i < 1
                    ),  # only first instance created from a video is marked as unique.
                    **kwargs,
                )

    def prepare_actions_field(
        self, steps: StepsT, add_noise: bool = False, task: Optional[Dict] = None
    ) -> Field:
        # We ignore candidates in this class but a child class can implement it
        steps_text = [step["desc"].strip() for step in steps]

        if add_noise and self.action_noise_type != "none":
            if self.action_noise_location == "random":
                for i in range(len(steps_text)):
                    if np.random.rand() < self.action_noise_replacement_prob:
                        steps_text[i] = np.random.choice(
                            task["action_descriptions"]
                            if self.action_noise_type == "random-in-task" and task
                            else self.file_reader.all_actions
                        )
            elif self.action_noise_location == "end":
                steps_text[-1] = np.random.choice(
                    task["action_descriptions"]
                    if self.action_noise_type == "random-in-task"
                    else self.file_reader.all_actions
                )
        action_tokens = [
            Token(
                text=step_text + self.end_of_action_text
                if self.end_of_action_text
                else step_text
            )
            for step_text in steps_text
        ]

        actions_field = TextField(action_tokens)
        return actions_field

    def prepare_task_field(self, task: Optional[Dict[str, Any]]) -> Field:
        assert task is not None
        task_field = TextField([Token(text=task["desc"])])
        return task_field

    def _pick_observations(
        self, steps: StepsT
    ) -> List[Union[torch.Tensor, np.ndarray]]:
        if self.file_reader.read_features:
            key_suffix = "_observation_features"
        else:
            key_suffix = "_observation_frames"

        if self.observation_type == "post":
            key = "post" + key_suffix
            observations = [self.image_features_transform(step[key]) for step in steps]
        elif self.observation_type == "pre":
            key = "pre" + key_suffix
            # ISSUE: This is a hack. The next actions pre is passed as the obs after current action.
            # Once the model can handle true "pre" observations. We need to change this.
            observations = [
                self.image_features_transform(step[key]) for step in steps[1:]
            ]
        else:
            raise ValueError
        obs = cast(
            List[Union[torch.Tensor, np.ndarray]], observations
        )  # to satisfy mypy
        return obs

    def prepare_observation_field(self, steps: StepsT, target: bool = False) -> Field:
        assert isinstance(self.file_reader, FileReader)
        obs_ = self._pick_observations(steps)
        if isinstance(obs_[0], np.ndarray):

            def stack(x):
                return np.stack(x, axis=0).astype(np.float32)

        else:

            def stack(x):
                return torch.stack(x, dim=0).to(dtype=torch.float32)

        observations = stack(
            obs_,
        )
        return ObservationFeaturesField(
            observations,
            min_sequence_padding_length=self.minimum_target_length if target else None,
        )  # Expected shape (num_steps, observation_length, image_feature_size) or (num_steps, C, H, W)

    def add_steps(
        self, steps: List[Dict[str, Union[float, str, None]]]
    ) -> List[Dict[str, Union[float, str, None]]]:
        steps = [step for step in steps if not step["artificial"]]
        if self.end_action_text:
            steps.append(
                asdict(
                    Step(
                        idx=0,
                        start=steps[-1]["end"],
                        end=steps[-1]["end"],
                        desc=self.end_action_text,
                        artificial=True,
                        pre_observation_frames=steps[-1]["post_observation_frames"],
                        post_observation_frames=steps[-1]["post_observation_frames"],
                        pre_observation_features=steps[-1]["post_observation_features"],
                        post_observation_features=steps[-1][
                            "post_observation_features"
                        ],
                        pre_observation_start=steps[-1]["post_observation_end"],
                        pre_observation_end=steps[-1]["post_observation_end"],
                        post_observation_start=steps[-1]["post_observation_end"],
                        post_observation_end=steps[-1]["post_observation_end"],
                    )
                )
            )
        if self.start_action_text:
            self.add_start_action(steps)
        return steps

    def add_start_action(self, steps: StepsT, reference: Step = None):
        if reference is None:
            assert len(steps) > 0
            reference = steps[0]

        steps.insert(
            0,
            asdict(
                Step(
                    idx=0,
                    start=0,
                    end=0,
                    desc=self.start_action_text,
                    artificial=True,
                    pre_observation_frames=reference["pre_observation_frames"],
                    post_observation_frames=reference["pre_observation_frames"],
                    pre_observation_features=reference["pre_observation_features"],
                    post_observation_features=reference["pre_observation_features"],
                    pre_observation_start=reference["pre_observation_start"],
                    pre_observation_end=reference["pre_observation_start"],
                    post_observation_start=reference["pre_observation_start"],
                    post_observation_end=reference["pre_observation_start"],
                )
            ),
        )

    def text_to_instance(  # type:ignore
        self,
        steps: StepsT,
        task_idx: Optional[str],
        task: Optional[Dict[str, Any]],
        prefix_start: int = 0,
        prefix_end: Optional[int] = None,
        unique_sequence: bool = True,
        **kwargs,
    ) -> Instance:
        assert isinstance(self.file_reader, FileReader)
        if "features_path" in kwargs:
            kwargs["features_path"] = str(kwargs["features_path"])
        if "video_path" in kwargs:
            kwargs["video_path"] = str(kwargs["video_path"])
        meta_dict: Dict = {**dict(task=task, steps=steps, task_idx=task_idx), **kwargs}
        main_fields: Dict[str, Field] = {
            "actions": self.prepare_actions_field(steps),
            "observations": self.prepare_observation_field(
                steps
                if self.observation_type == "post"
                else (steps + [steps[-1]])
                if self.observation_type == "pre"
                else _raise_value_error()
            ),  # Expected shape (num_steps, observation_length, image_feature_size) or (num_steps, C, H, W)
            "unique_sequence": ArrayField(np.array(unique_sequence), dtype=np.bool8),
        }
        if self.create_task_field:
            main_fields["task"] = self.prepare_task_field(task)
        if prefix_end is not None:
            meta_dict["prefix_end"] = prefix_end
            if self.file_reader.segments_file is None:
                action_prefix = steps[prefix_start:prefix_end]
            else:
                _action_prefix = self.file_reader.get_steps_from_segments(
                    kwargs["idx"], int(floor(steps[prefix_end]["start"]))
                )
                self.file_reader.add_observation_features_to_steps(
                    _action_prefix, kwargs["idx"]
                )
                action_prefix = [asdict(step) for step in _action_prefix]
                if self.start_action_text:
                    self.add_start_action(action_prefix, reference=steps[prefix_end])

            prefix_length = prefix_end - prefix_start

            # Bring the last token in prefix at the beginning
            if self.move_last_prefix_action_to_begining:
                action_prefix.insert(0, action_prefix.pop(-1))
            action_target = steps[prefix_end:]
            meta_dict["prefix_length"] = prefix_length
            meta_dict["target_length"] = len(action_target)
            main_fields.update(
                {
                    "prefix_observations": self.prepare_observation_field(
                        action_prefix
                        if self.observation_type == "post"
                        else (action_prefix + [action_target[0]])
                        if self.observation_type == "pre"
                        else _raise_value_error()
                    ),
                    "prefix_actions": self.prepare_actions_field(
                        action_prefix, add_noise=True, task=task
                    ),
                    "target_observations": self.prepare_observation_field(
                        action_target
                        if self.observation_type == "post"
                        else (action_target + [action_target[-1]])
                        if self.observation_type == "pre"
                        else _raise_value_error(),
                        target=True,
                    ),
                    "target_actions": self.prepare_actions_field(
                        action_target, add_noise=False
                    ),
                    "prefix_length": ArrayField(
                        torch.tensor(prefix_length, dtype=torch.long)
                    ),
                }
            )

        return Instance({**main_fields, "meta": MetadataField(meta_dict)})

    def apply_token_indexers(self, instance: Instance) -> None:
        instance["actions"].token_indexers = self.action_token_indexers  # type: ignore
        instance["task"].token_indexers = self.task_token_indexers  # type: ignore
        if self.create_prefixes:
            instance["prefix_actions"].token_indexers = self.action_token_indexers  # type: ignore
            instance["target_actions"].token_indexers = self.target_action_token_indexers  # type: ignore

    def _read(self, file_path: str) -> Iterator[Instance]:
        file_path_ = Path(file_path)
        # ISSUE: data_path and split_file are not requirements of the base FileReader
        #   but are specific to crosstask file reader!
        self.file_reader.set_split_file_and_data_path(
            data_path=file_path_.parent, split_file=file_path_.name
        )
        assert isinstance(self.file_reader, FileReader)
        skip_count = 0
        try:
            for annotated_video in self.file_reader.read(sharder=self.shard_iterable):
                for instance in self.text_to_instances(**asdict(annotated_video)):
                    target_length = instance["meta"].get("target_length", 10000000)
                    prefix_length = instance["meta"].get("prefix_length", 10000000)
                    if (
                        target_length >= self.minimum_target_length
                        and prefix_length >= self.minimum_prefix_length
                    ):
                        yield instance
                    else:
                        skip_count += 1
        finally:
            logger.info(f"{skip_count} instances skipped due to short length.")
