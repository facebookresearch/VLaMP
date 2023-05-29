# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


"""Modules common for all the datasets"""
from genericpath import isfile
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from re import S
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    TypeVar,
    Union,
)
import PIL
from allennlp.common.registrable import Registrable
from git import PathLike
import numpy as np
import torch
import random

torch.multiprocessing.set_sharing_strategy("file_system")
from wcmatch import glob
import torchvision.transforms.functional as VF
from torchvision.io import read_video
from vlamp.flags import DEBUG

_T = TypeVar("_T")

SharderT = Callable[[Iterable[_T]], Iterable[_T]]


logger = logging.getLogger(__name__)

Pathlike = Union[Path, str]

SplitT = Literal["train", "validation", "test"]

ActionNoiseT = Literal["none", "random", "random-in-task"]


class TensorCacheDir:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _file(self, key: str) -> Path:
        file = (self.path / key).with_suffix(".pt")
        return file

    def get(self, key: str, default: None = None) -> Optional[torch.Tensor]:
        file = self._file(key)
        if file.is_file():
            t = torch.load(file)
        else:
            t = default
        return t

    def set(self, key: str, value: torch.Tensor) -> None:
        file = self._file(key)
        file.parent.mkdir(exist_ok=True, parents=True)
        torch.save(value, file)


@dataclass
class Step:
    idx: str
    start: float
    end: float
    artificial: bool = False  # :> not in data but added artificially for some purpose
    desc: Optional[str] = None
    pre_observation_start: Optional[int] = None  # :> previous obs
    pre_observation_end: Optional[int] = None
    pre_observation_features: Optional[np.ndarray] = None
    pre_observation_frames: Optional[
        Union[List[PIL.Image.Image], List[torch.Tensor], List[np.ndarray]]
    ] = None
    post_observation_start: Optional[int] = None  # :> previous obs
    post_observation_end: Optional[int] = None
    post_observation_features: Optional[np.ndarray] = None
    post_observation_frames: Optional[
        Union[List[PIL.Image.Image], List[torch.Tensor], List[np.ndarray]]
    ] = None
    observation_feature_indices: Optional[List[int]] = None


@dataclass
class AnnotatedVideo:
    idx: Optional[str] = None
    steps: List[Step] = field(default_factory=list)
    task: Optional["Task"] = None
    task_idx: Optional[str] = None  # either specify task or task_id
    features: Optional[np.ndarray] = None  # :> all features for the video
    frame_times: Optional[
        np.ndarray
    ] = None  # :> timestamp in seconds corresponding to the features
    frames: Optional[
        np.ndarray
    ] = None  # :> actual image frames corresponding to the features
    features_path: Optional[Pathlike] = None  # path to features on the disk
    video_path: Optional[Pathlike] = None  # Path to the video on the disk
    steps_from_segmentation: Optional[List[Step]] = None


@dataclass
class Task:
    idx: str
    desc: str = ""
    url: Optional[str] = None  #: wikihow url for crosstask
    num_steps: Optional[int] = None
    stepid2desc: Mapping[str, str] = field(default_factory=dict)
    action_descriptions: List[str] = field(default_factory=list)
    annotated_videos: List[AnnotatedVideo] = field(default_factory=list)

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Task):
            return self.idx == __o.idx
        return False


def add_step_descriptions(
    annotated_video: AnnotatedVideo,
    task: Optional[Task] = None,
    prefix_task_description: bool = False,
) -> AnnotatedVideo:
    if task is not None:
        if annotated_video.task is not None:
            assert annotated_video.task.idx == task.idx
    else:
        assert annotated_video.task is not None
        task = annotated_video.task
    assert task.desc
    for step in annotated_video.steps:
        if prefix_task_description:
            step.desc = f"{task.desc}: {task.stepid2desc[step.idx]}"
        else:
            step.desc = task.stepid2desc[step.idx]
    return annotated_video


class FileReader(Registrable):
    videos_folder = "videos"

    def __init__(
        self,
        data_path: Optional[Pathlike] = None,
        split_file: Optional[Pathlike] = None,
        prefix_task_descriptions_to_steps: bool = False,
        read_features: bool = False,
        read_frames: bool = False,
        observation_length: int = 2,
        observation_offset: int = -1,
        debug: bool = DEBUG,
        segments_file: Optional[Pathlike] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.data_path: Path = Path(data_path) if data_path is not None else None
        self.split: Optional[Set[str]] = None
        self.prefix_task_descriptions_to_steps = prefix_task_descriptions_to_steps
        self.read_features = read_features
        self.read_frames = read_frames
        self.observation_length = observation_length
        self.observation_offset = observation_offset
        self.split_file = None
        if split_file is not None:
            self.set_split_file(split_file)
        self.debug = debug
        self.tensor_cache = None
        self._all_actions: Optional[List] = None
        self.feature_indices_to_take = None
        self.features_folder = "features"
        self.segments_file = segments_file
        self._segments: Optional[Dict[str, List[str]]] = None
        self.minimum_segment_length = 2  # seconds
        self.null_action = "O"

    def set_split_file_and_data_path(
        self, data_path: Path, split_file: PathLike
    ) -> None:
        self.data_path = data_path
        self.set_split_file(split_file)
        self.setup_tensor_cache()

    @property
    def segments(self) -> Optional[Dict[str, List[str]]]:
        if self._segments is None and self.segments_file is not None:
            self._segments = self.read_segments_file(
                self.data_path / self.segments_file
            )
        return self._segments

    def read_segments_file(self, segments_file: PathLike) -> Dict[str, List[str]]:
        with open(segments_file, "r") as f:
            segments = {vid["idx"]: vid["y_pred"] for vid in json.load(f)}
        return segments

    def get_steps_from_segments(self, video_idx: str, end: int = None) -> List[Step]:
        assert self.segments is not None
        segments = self.segments.get(video_idx, None)
        if segments is None:
            raise ValueError(f"Video {video_idx} not found in segments file")
        return self.segments_to_steps(self.segments[video_idx], end)

    def segments_to_steps(self, segments: List[str], end: int = None) -> List[Step]:
        steps: List[Step] = []
        start_t = 0
        start_frame_action = segments[start_t]
        if end is None:
            end = len(segments)
        force_end = False
        for t, frame_action in enumerate(segments[1:], start=1):
            if t <= end:
                if t == end:
                    force_end = True
                if start_frame_action != frame_action or force_end:
                    if (
                        t - start_t >= self.minimum_segment_length
                        and start_frame_action != self.null_action
                    ):
                        steps.append(
                            Step(
                                idx="null",
                                start=start_t,
                                end=t - 1,
                                desc=start_frame_action,
                            )
                        )
                    start_t = t
                    start_frame_action = frame_action
            else:
                break
        return steps

    @property
    def all_actions(self) -> List[str]:
        if self._all_actions is None:
            self._all_actions = list(
                set([a for t in self.tasks for a in t.stepid2desc.values()])
            )
        return self._all_actions

    def setup_tensor_cache(self) -> None:
        assert self.data_path is not None
        if self.read_frames:
            # self.tensor_cache = TensorCache(self.data_path / "frames_tensor_cache")
            self.tensor_cache = TensorCacheDir(
                self.data_path / "frames_tensor_cache_dir"
            )
        else:
            self.tensor_cache = None

    def set_split_file(self, split_file) -> None:
        if split_file is not None:
            split_ = []
            for file_ in glob.glob(
                str(self.data_path / split_file), flags=glob.EXTGLOB | glob.BRACE
            ):
                with open(file_) as f:
                    for line in f.readlines():
                        split_.append(line.strip())
            self.split = set(split_)  # example ids in the split
        self.split_file = split_file

    def in_split(self, annotated_video: AnnotatedVideo) -> bool:
        if self.split is not None:
            assert (
                annotated_video.idx is not None
            ), f"Split is set but video {annotated_video} does not have id"
            return annotated_video.idx in self.split
        return True

    @property
    def tasks(self) -> List[Task]:
        raise NotImplementedError

    @property
    def annotations(self) -> List[AnnotatedVideo]:
        raise NotImplementedError

    def read(self, sharder: SharderT) -> Iterator[AnnotatedVideo]:
        raise NotImplementedError

    def get_pil_image_frames(
        self,
        video_idx: str,
        start: float,
        end: float,
    ) -> List[PIL.Image.Image]:
        video_file = self.get_video_path(video_idx)
        video, audio, metadata = read_video(
            str(video_file), start_pts=start, end_pts=end, pts_unit="sec"
        )
        # video shape: (num_frames, )
        images = [VF.to_pil_image(image) for image in video.numpy()]
        return images

    def read_video_frames_as_tensor(self, video_idx: str) -> torch.Tensor:
        video_file = self.get_video_path(video_idx)
        video, audio, metadata = read_video(
            str(video_file),
            pts_unit="sec",
        )
        video = video.permute(0, 3, 1, 2)  # TCHW
        return video

    def get_image_frame_as_tensor(
        self,
        video_idx: str,
        time: float,
    ) -> torch.Tensor:
        key = f"{video_idx}_{int(time)}"
        frame = None
        if self.tensor_cache is not None:
            frame = self.tensor_cache.get(key, None)
        if frame is None:
            video_file = self.get_video_path(video_idx)
            video, audio, metadata = read_video(
                str(video_file),
                start_pts=time,
                end_pts=time + 0.0001,
                pts_unit="sec",
            )
            # video shape: (num_frames, )
            video = video.permute(0, 3, 1, 2)
            frame = video[0].contiguous()

            if self.tensor_cache is not None:
                # self.tensor_cache[key] = frame
                self.tensor_cache.set(key, frame)
        assert frame is not None
        return frame

    def get_features_file(self, video_idx: str) -> Path:
        return self.data_path / self.features_folder / f"{video_idx}.npy"

    @staticmethod
    def _read_features(
        features_folder: Path,
        features_to_take: Optional[List[int]] = None,
        annotated_video: Optional[AnnotatedVideo] = None,
        video_idx: Optional[str] = None,
    ) -> np.ndarray:
        """_summary_

        Args:
            annotated_video (AnnotatedVideo): _description_
            features_folder (Path): _description_
            features_to_take: Indices of the features to take.

        Returns:
            AnnotatedVideo: _description_
        """
        if annotated_video is None:
            assert video_idx is not None
            annotated_video = AnnotatedVideo(idx=video_idx)
        assert annotated_video.idx is not None
        annotated_video.features_path = (
            features_folder / annotated_video.idx
        ).with_suffix(".npy")
        if not annotated_video.features_path.exists():
            raise MissingFeaturesException(
                f"Features file {annotated_video.features_path} is missing."
            )
        features = np.load(annotated_video.features_path)
        assert features is not None
        if features_to_take is not None:
            features = features[..., features_to_take]
        return features

    def get_video_path(
        self, annotated_video_or_video_id: Union[AnnotatedVideo, str]
    ) -> Path:
        if isinstance(annotated_video_or_video_id, AnnotatedVideo):
            return (
                self.data_path
                / self.videos_folder
                / f"{annotated_video_or_video_id.idx}.mp4"
            )
        else:
            return (
                self.data_path
                / self.videos_folder
                / f"{annotated_video_or_video_id}.mp4"
            )

    def add_observation_features_to_steps(
        self, steps: List[Step], video_idx: str
    ) -> List[Step]:
        all_features = self._read_features(
            video_idx=video_idx,
            features_folder=self.data_path / self.features_folder,
            features_to_take=self.feature_indices_to_take,
        )
        max_index = all_features.shape[0] - 1

        def bound(index: Union[int, float]) -> int:
            return int(max(0, min(max_index, index)))

        for step in steps:
            step.pre_observation_start = int(
                np.floor(step.start + self.observation_offset)
            )

            step.pre_observation_end = (
                step.pre_observation_start + self.observation_length
            )
            step.post_observation_start = int(
                np.floor(step.end - self.observation_offset)
            )
            # offset is set using start so negative value has to be added for end
            step.post_observation_end = (
                step.post_observation_start + self.observation_length
            )
            step.pre_observation_features = all_features[
                [
                    bound(t)
                    for t in range(step.pre_observation_start, step.pre_observation_end)
                ]
            ]
            step.post_observation_features = all_features[
                [
                    bound(t)
                    for t in range(
                        step.post_observation_start, step.post_observation_end
                    )
                ]
            ]
        return steps

    def add_observation_features_and_frames(
        self, annotation: AnnotatedVideo
    ) -> AnnotatedVideo:
        assert annotation.idx is not None
        missing_video = not self.get_video_path(annotation).exists()
        missing_features = False
        video_reading_error = ""
        if self.read_features:
            try:
                all_features = self._read_features(
                    annotated_video=annotation,
                    features_folder=self.data_path / self.features_folder,
                    features_to_take=self.feature_indices_to_take,
                )
                max_index = all_features.shape[0] - 1
            except MissingFeaturesException as e:
                missing_features = True
                max_index = 99999999999
        else:
            max_index = 99999999999
        assert annotation.steps is not None

        def bound(index: Union[int, float]) -> int:
            return int(max(0, min(max_index, index)))

        for step in annotation.steps:
            step.pre_observation_start = int(
                np.floor(step.start + self.observation_offset)
            )

            step.pre_observation_end = (
                step.pre_observation_start + self.observation_length
            )
            step.post_observation_start = int(
                np.floor(step.end - self.observation_offset)
            )
            # offset is set using start so negative value has to be added for end
            step.post_observation_end = (
                step.post_observation_start + self.observation_length
            )
            if self.read_features and not missing_features:
                step.pre_observation_features = all_features[
                    [
                        bound(t)
                        for t in range(
                            step.pre_observation_start, step.pre_observation_end
                        )
                    ]
                ]
                step.post_observation_features = all_features[
                    [
                        bound(t)
                        for t in range(
                            step.post_observation_start, step.post_observation_end
                        )
                    ]
                ]
            if self.read_frames and not missing_video:
                try:
                    step.pre_observation_frames = [
                        self.get_image_frame_as_tensor(
                            annotation.idx,
                            bound(
                                int(
                                    (
                                        step.pre_observation_start
                                        + step.pre_observation_end
                                    )
                                    / 2
                                )
                            ),
                        )
                    ]
                    step.post_observation_frames = [
                        self.get_image_frame_as_tensor(
                            annotation.idx,
                            bound(
                                int(
                                    (
                                        step.post_observation_start
                                        + step.post_observation_end
                                    )
                                    / 2
                                )
                            ),
                        )
                    ]
                except Exception as e:
                    video_reading_error = str(e)

        if self.read_features and missing_features:
            raise MissingFeaturesException(
                f"Features file {annotation.features_path} is missing."
            )
        if self.read_frames and missing_video:
            raise MissingVideoFileException(
                f"Video file {annotation.video_path} is missing."
            )
        if self.read_frames and video_reading_error:
            raise MissingVideoFileException(
                f"Error {video_reading_error} reading video {annotation.video_path}."
            )

        return annotation


class DataReadingException(Exception):
    pass


class MissingFeaturesException(DataReadingException):
    pass


class MissingVideoFileException(DataReadingException):
    pass
