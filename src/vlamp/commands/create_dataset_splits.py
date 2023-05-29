# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


"""
Sample subcommand.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import random
import math
from typing import List


from allennlp.commands.subcommand import Subcommand
from allennlp.common.params import Params
from vlamp.dataset_readers.common import AnnotatedVideo, FileReader


logger = logging.getLogger(__name__)

RANDOM_SEED = 10
logger.info(f"Random seed: {RANDOM_SEED}")


@Subcommand.register("create-dataset-splits")
class CreateDatasetSplits(Subcommand):
    def add_subparser(
        self, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        description = """Split into training, validation and test data."""
        subparser = parser.add_parser(
            self.name, description=description, help=description
        )
        subparser.add_argument("data_path", type=Path, help="path to the data folder")
        subparser.add_argument(
            "--train-percent",
            type=float,
            default=70,
            help="Size in percentage of train split. The rest is left for test.",
        )
        subparser.add_argument(
            "--validation-percent",
            type=float,
            default=20,
            help="Size in percent of the train set to be used for validation (default: 20)",
        )
        subparser.add_argument(
            "-s",
            "--shuffle",
            action="store_true",
            help="Shuffle before splitting (default True)",
        )
        subparser.add_argument(
            "-r",
            "--file-reader",
            type=str,
            required=True,
            choices=["crosstask", "coin"],
            help='Name of the file reader (default: "crosstask")',
        )

        subparser.set_defaults(func=split)

        return subparser


def split(args: argparse.Namespace):

    split_data(
        args.data_path, args.file_reader, args.train_percent, args.validation_percent
    )


def split_data(
    data_path: Path, file_reader: str, train_percent: float, validation_percent: float
):
    """Split the data into train,val and test

    Args:
        data_path (Path): Path to the folder containing data
        file_reader (str): registered name of the file reader. For example, crosstask or coin.
        train_percent (float): Percent to keep for train set
        validation_percent (float): Percent of train set to keep for validation
    """
    reader = FileReader.from_params(
        Params({"type": file_reader, "data_path": data_path})
    )
    annotations_ = reader.annotations
    # Take only those annotations that have video features
    annotations = [ann for ann in annotations_ if reader.get_features_file(ann.idx).is_file()]
    random.seed(RANDOM_SEED)
    random.shuffle(annotations)
    train_split_index = math.ceil(train_percent / 100.0 * len(annotations))
    train = annotations[:train_split_index]
    test = annotations[train_split_index:]
    logger.info(f"Total instances: {len(annotations)}")
    val: List[AnnotatedVideo] = []
    if validation_percent:
        val_split_index = math.ceil(validation_percent / 100.0 * len(train))
        val = train[:val_split_index]
        train = train[val_split_index:]
    for split_name, split in [("train", train), ("validation", val), ("test", test)]:
        logger.info(f"Number of {split_name} instances: {len(split)}")
        output_file = (data_path / split_name).with_suffix(".txt")
        logger.info(f"Writing {split_name} to {output_file}")
        with open(output_file, "w") as f:
            a: AnnotatedVideo
            for a in split:
                f.write(f"{a.idx}\n")
