# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import json
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from allennlp.common.util import A


from .common import AnnotatedVideo, FileReader, Step, Task, SharderT
import logging

logger = logging.getLogger(__name__)


@FileReader.register("coin")
class CoinReader(FileReader):
    json_path = "COIN.json"
    s3d_features_folder = "s3dg_features"
    videos_folder = "videos"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._tasks: Optional[List[Task]] = None
        self._annotated_videos: Optional[List[AnnotatedVideo]] = None
        self.features_folder = self.s3d_features_folder

    @property
    def tasks(self) -> List[Task]:
        if self._tasks is None:
            assert self._annotated_videos is None, "Mixed read"
            assert self._tasks is None, "Mixer read"
            self._tasks, self._annotated_videos = self.read_annotation_file(
                self.data_path / self.json_path
            )
        return self._tasks

    @property
    def annotations(self) -> List[AnnotatedVideo]:
        if self._annotated_videos is None:
            assert self._tasks is None, "Mixed read"
            self._tasks, self._annotated_videos = self.read_annotation_file(
                self.data_path / self.json_path
            )
        return self._annotated_videos

    def read(self, sharder: SharderT) -> Iterator[AnnotatedVideo]:
        tasks, annotations = self.read_annotation_file(self.data_path / self.json_path)

        def in_split(
            annotations_: Iterable[AnnotatedVideo],
        ) -> Iterable[AnnotatedVideo]:
            assert self.split is not None
            for annotation in annotations_:
                if annotation.idx in self.split:
                    yield annotation

        for annotated_video in sharder(in_split(annotations)):
            self.add_observation_features_and_frames(annotated_video)
            yield annotated_video

    def read_annotation_file(
        self, filepath: Path
    ) -> Tuple[List[Task], List[AnnotatedVideo]]:
        with open(filepath, "r") as f:
            data = json.load(f)["database"]

        annotated_videos_dict: Dict[str, AnnotatedVideo] = {}
        tasks_dict: Dict[str, Task] = {}
        logger.debug("Reading tasks and annotations for COIN.")
        # read
        for example_id, example in data.items():
            task_id = str(example["recipe_type"])
            # Create new task if seeing it for the first time
            if task_id not in tasks_dict:
                tasks_dict[task_id] = Task(task_id, desc=example["class"])
            # Read annotated video
            if example_id not in annotated_videos_dict:
                annotated_video = AnnotatedVideo(
                    idx=example_id,
                    steps=[
                        Step(
                            idx=_step["id"],
                            start=float(_step["segment"][0]),
                            end=float(_step["segment"][1]),
                            desc=_step["label"],
                        )
                        for _step in example["annotation"]
                    ],
                    task=tasks_dict[task_id],
                )
                # Add other things
                annotated_video.video_path = self.get_video_path(annotated_video)
                # See if there are any new actions for the task in this example
                for step in annotated_video.steps:
                    if step.idx not in tasks_dict[task_id].stepid2desc:
                        tasks_dict[task_id].stepid2desc[step.idx] = step.desc.strip()
                    assert (
                        tasks_dict[task_id].stepid2desc[step.idx] == step.desc.strip()
                    )
                annotated_videos_dict[example_id] = annotated_video
                # We will not add annotated videos to the task, as it will create stack overflow
                # when converting to dict in the dataset reader.
        # update the num_steps attribute
        for task_id, task in tasks_dict.items():
            task.num_steps = len(task.stepid2desc)
        return list(tasks_dict.values()), list(annotated_videos_dict.values())
