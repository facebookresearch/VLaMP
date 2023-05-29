# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from __future__ import annotations
from itertools import chain
from pathlib import Path
import re
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    TextIO,
    Tuple,
    TypeVar,
    Union,
)
import numpy as np
import torch
import tqdm
from transformers import CLIPProcessor, CLIPVisionModel

from .common import (
    DataReadingException,
    FileReader,
    MissingFeaturesException,
    MissingVideoFileException,
    Task,
    AnnotatedVideo,
    Step,
    add_step_descriptions,
    SharderT,
)
import logging


logger = logging.getLogger(__name__)

ObservationLocationT = Literal["start", "end"]


@FileReader.register("crosstask")
class CrosstaskReader(FileReader):
    _task_file = "tasks_primary.txt"
    annotations_folder = "annotations"
    i3d_features_folder = "features"
    s3d_features_folder = "s3dg_features"
    videos_folder = "videos"
    ANNOTATION_FILE_REGEX = r"^(\d+)_([\w-]+)\.csv$"
    i3d_features_range = (0, 1024)
    resnet_features_range = (1024, 1024 + 2048)
    audio_features_range = (1024 + 2048, 1024 + 2048 + 128)
    s3d_features_range = (0, 1024)

    def __init__(
        self,
        data_path: Optional[Path] = None,
        s3d_features: bool = True,
        i3d_features: bool = False,
        resnet_features: bool = False,
        audio_features: bool = False,
        add_extra_steps: bool = False,
        task_file: Optional[Path] = None,
        **kwargs,
    ) -> None:
        super().__init__(data_path=data_path, **kwargs)
        self._tasks: Optional[List[Task]] = None
        self._annotated_videos: Optional[List[AnnotatedVideo]] = None

        self.task_file = task_file or self._task_file
        if i3d_features or resnet_features or audio_features:
            self.features_folder = self.i3d_features_folder
            if i3d_features and resnet_features and audio_features:
                self.feature_indices_to_take = None
            else:
                feature_indices_to_take: List[int] = []
                if i3d_features:
                    feature_indices_to_take += list(
                        range(*CrosstaskReader.i3d_features_range)
                    )
                if resnet_features:
                    feature_indices_to_take += list(
                        range(*CrosstaskReader.resnet_features_range)
                    )
                if audio_features:
                    feature_indices_to_take += list(
                        range(*CrosstaskReader.audio_features_range)
                    )
                self.feature_indices_to_take = feature_indices_to_take
            self.num_features = (
                len(self.feature_indices_to_take)
                if self.feature_indices_to_take is not None
                else CrosstaskReader.audio_features_range[-1]
            )
        else:  # s3d_features = True
            self.features_folder = self.s3d_features_folder
            self.feature_indices_to_take = None
            self.num_features = self.s3d_features_range[-1]

        self.add_extra_steps = add_extra_steps

    @property
    def tasks(self) -> List[Task]:
        if self._tasks is None:
            self._tasks = self.read_tasks_file(self.data_path / self.task_file)
        return self._tasks

    @property
    def annotations(self) -> List[AnnotatedVideo]:
        if self._annotated_videos is None:
            _ = self.tasks  # Make sure tasks have been read
            self._annotated_videos = [
                ann
                for task in self.tasks
                for ann in self.read_annotations(
                    self.data_path / self.annotations_folder, task=task
                )
                if self.in_split(
                    ann
                )  # if the split (train/val/test) is set, only read those
            ]
        return self._annotated_videos

    @staticmethod
    def _read_task(f: TextIO) -> Optional[Task]:
        task_id = f.readline().strip()
        if task_id:  # not end of file
            task_desc = f.readline().strip()
            task_url = f.readline().strip()
            task_num_steps = int(f.readline().strip())
            step_descs = f.readline().strip().split(",")
            assert task_num_steps == len(step_descs)
            empty_line = f.readline()

            return Task(
                idx=task_id,
                desc=task_desc,
                url=task_url,
                num_steps=task_num_steps,
                action_descriptions=step_descs,
                stepid2desc={
                    str(i): step_desc for i, step_desc in enumerate(step_descs, 1)
                },
            )
        else:
            return None

    @staticmethod
    def read_tasks_file(filepath) -> List[Task]:
        tasks = []
        with open(filepath) as f:
            while True:
                task = CrosstaskReader._read_task(f)
                if task is None:
                    break
                else:
                    tasks.append(task)
        return tasks

    @staticmethod
    def read_annotation_file(
        filepath: Path, task: Optional[Task] = None
    ) -> AnnotatedVideo:
        annotated_video = AnnotatedVideo(
            idx=CrosstaskReader.get_video_id_from_filename(filepath)
        )
        if task:
            task.annotated_videos.append(annotated_video)
        annotated_video.task = task
        if task is not None:
            annotated_video.task_idx = task.idx
        with open(filepath) as f:
            for line in f:
                if line:
                    idx, start, end = [
                        component.strip() for component in line.split(",")
                    ]  # strip new lines and spaces
                    annotated_video.steps.append(Step(idx, float(start), float(end)))
        return annotated_video

    @staticmethod
    def read_annotations(
        folderpath: Path, task: Optional[Task] = None
    ) -> List[AnnotatedVideo]:
        files = tqdm.tqdm(
            folderpath.glob("*.csv")
            if task is None
            else folderpath.glob(f"{task.idx}_*.csv"),
            desc=(
                f"Reading annotations for crosstask for task {task.desc or task.idx}"
                if task
                else "Reading annotations for crosstask"
            ),
        )
        if task is not None:
            if task.annotated_videos:
                logger.warning(
                    f"{task.desc} already has annotated videos. Emptying it before adding again."
                )
        return [
            CrosstaskReader.read_annotation_file(a_file, task=task) for a_file in files
        ]

    def annotation_files_iterator(self, folderpath: Path) -> Iterable[Path]:
        assert self.split is not None
        for a_file in sorted(folderpath.glob("*.csv")):
            if self.get_video_id_from_filename(a_file) in self.split:
                yield a_file

    def read(
        self,
        sharder: SharderT,
    ) -> Iterator[AnnotatedVideo]:
        folderpath = self.data_path / self.annotations_folder
        task_dict: Dict[str, Task] = {task.idx: task for task in self.tasks}
        logger.info("Reading the following tasks")
        for t, _ in task_dict.items():
            logger.info(f"{t}: {_.desc}")
        assert self.split is not None
        for a_file in sharder(self.annotation_files_iterator(folderpath)):
            logger.debug(f"reading {a_file}")
            task_idx = CrosstaskReader.get_task_from_filename(a_file).idx
            if task_idx not in task_dict:  # skip the task
                continue
            annotation = CrosstaskReader.read_annotation_file(a_file)
            annotation.task_idx = task_idx
            annotation.task = task_dict[annotation.task_idx]
            annotation.video_path = self.get_video_path(annotation)
            annotation = add_step_descriptions(
                annotation,
                prefix_task_description=self.prefix_task_descriptions_to_steps,
            )
            try:
                self.add_observation_features_and_frames(annotation)
            except DataReadingException as e:
                logger.error(e)
                continue

            yield annotation

    @staticmethod
    def get_task_from_filename(filename: Path) -> Task:
        """Get task name from the filepath.

        Note: The name of the file is not verified and is assumed to be in the expected format.

        Args:
            filename (Path): File in the annotations folder. The name should have the form <task-id>__<vid-id>.csv

        Returns:
            Task: task
        """
        m = re.fullmatch(CrosstaskReader.ANNOTATION_FILE_REGEX, filename.name)
        if m:
            return Task(idx=m[1])
        else:
            IOError(f"Invalid filename {filename}")

    @staticmethod
    def get_video_id_from_filename(filename: Path) -> str:
        """Get video id from the name of the annotation file.

        Args:
            filename (Path):

        Returns:
            str: _description_
        """
        m = re.fullmatch(CrosstaskReader.ANNOTATION_FILE_REGEX, filename.name)
        if m:
            return m[2]
        else:
            raise IOError(f"Invalid filename {filename}")


@FileReader.register("crosstask-with-clip")
class CrosstaskReaderWithCLIP(CrosstaskReader):
    clip_features_folder = "clip_features"

    def __init__(
        self,
        data_path: Optional[Path] = None,
        observation_length: int = 2,
        observation_offset: int = -1,
        s3d_features: bool = True,
        i3d_features: bool = False,
        resnet_features: bool = False,
        audio_features: bool = False,
        add_extra_steps: bool = False,
        task_file: Optional[Path] = None,
        clip_model: str = "openai/clip-vit-base-patch32",
        **kwargs,
    ) -> None:
        super().__init__(
            data_path,
            observation_length,
            observation_offset,
            s3d_features,
            i3d_features,
            resnet_features,
            audio_features,
            add_extra_steps,
            task_file,
            **kwargs,
        )
        self.clip_model_name = clip_model
        self.preprocessor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.embedder = CLIPVisionModel.from_pretrained(self.clip_model_name)
        self.embedder.requires_grad_(False)

    def get_feature_cache_file(self, video_idx: str, time: int) -> Path:
        features_path = (
            self.data_path / self.clip_features_folder / video_idx / str(time)
        ).with_suffix(".pt")
        features_path.parent.mkdir(parents=True, exist_ok=True)
        return features_path

    def get_encoded_features(self, idx: str, start: int, end: int) -> torch.Tensor:
        encode = False
        cache_files = []
        for i, time in enumerate(range(start, end)):
            cache_files.append(self.get_feature_cache_file(idx, time))
            if not cache_files[-1].exists():
                encode = True

        if encode:
            try:
                frames = [
                    self.get_image_frame_as_tensor(idx, time)
                    for i, time in enumerate(range(start, end))
                ]
            except Exception as e:
                raise MissingVideoFileException(str(e)) from Exception
            preprocessed = self.preprocessor(images=frames, return_tensors="pt")[
                "pixel_values"
            ]
            features = self.embedder(
                preprocessed
            ).last_hidden_state  # (T, num_clip_tokens, clip_hidden_size)
            T, num_clip_tokens, clip_hidden_size = features.shape
            for i, cache_file in enumerate(cache_files):
                torch.save(features[i], cache_file)
            features = features.reshape(T * num_clip_tokens, clip_hidden_size)

        else:
            features = torch.stack([torch.load(cf) for cf in cache_files], dim=0)
            T, num_clip_tokens, clip_hidden_size = features.shape
            features = features.reshape(T * num_clip_tokens, clip_hidden_size)

        return features

    def add_observation_features_and_frames(
        self, annotation: AnnotatedVideo
    ) -> AnnotatedVideo:
        assert annotation.idx is not None
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

            step.pre_observation_features = self.get_encoded_features(
                annotation.idx, step.pre_observation_start, step.pre_observation_end
            )
            step.post_observation_features = self.get_encoded_features(
                annotation.idx,
                step.post_observation_start,
                step.post_observation_end,
            )

        return annotation
