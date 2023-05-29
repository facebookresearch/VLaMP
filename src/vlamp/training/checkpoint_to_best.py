# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import os
from typing import Any, Dict
from allennlp.common.file_utils import hardlink_or_copy
from allennlp.training.callbacks import TrainerCallback
import logging

logger = logging.getLogger(__name__)


@TrainerCallback.register("checkpoint_to_best")
class CheckpointToBestCallback(TrainerCallback):
    def on_end(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any] = None,
        epoch: int = None,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        last_checkpoint = trainer._checkpointer.find_latest_checkpoint()
        model_state_file, _ = last_checkpoint
        if trainer._best_model_filename is not None and os.path.exists(
            trainer._best_model_filename
        ):
            logger.info(f"{trainer._best_model_filename} exits already. Doing nothing.")
            pass
        else:
            trainer._best_model_filename = os.path.join(
                trainer._serialization_dir, "best.th"
            )
            hardlink_or_copy(model_state_file, trainer._best_model_filename)
            logger.info(f"saving {model_state_file} as {trainer._best_model_filename}")
