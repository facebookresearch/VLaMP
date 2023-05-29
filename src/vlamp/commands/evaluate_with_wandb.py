# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import pathlib
import tempfile
from typing import Tuple, List, Dict, Any, Optional, Union
from wandb_allennlp.commands.parser_base import (
    SetWandbEnvVar,
    WandbParserBase,
    read_from_env,
)
from allennlp.commands import Subcommand
import argparse
import logging
import json
import os
import sys
from wandb_allennlp.commands.train_with_wandb import translate
from allennlp.evaluation.evaluator import DataLoader, Evaluator
from allennlp.common import logging as common_logging
from pathlib import Path
from allennlp.models.archival import load_archive
from copy import deepcopy
from allennlp.common.util import prepare_environment
from json import JSONDecodeError

logger = logging.getLogger(__name__)

MODEL_FILE = "model.tar.gz"


@Subcommand.register("evaluate-with-wandb")
class EvaluateWithWandb(WandbParserBase):
    description = "Evaluate with logging to wandb"
    help_message = (
        "Use `allennlp evaluate-with-wandb` subcommand instead of "
        "`allennp evaluate` to log eval results to wandb. "
        "It supports all the arguments present in `allennlp evaluate`. "
        "However, the --overrides have to be specified in the `--kw value` or `--kw=value` form, "
        "where 'kw' is the parameter to override and 'value' is its value. "
        "Use the dot notation for nested parameters. "
        "For instance, {'model': {'embedder': {'type': xyz}}} can be provided as --model.embedder.type xyz"
    )
    require_run_id = False

    def add_arguments(
        self, subparser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        # we use the same args as the allennlp evaluate command
        # except the --overrides
        # and input_path because
        # overrides is something we will create
        # and param_path is not a kwarg and hence is always required
        # We cannot have a compulsory arg here because if we do and
        # we are not trying to call train_with_wandb but some other command
        # The feeler call to parse_know_args() will throw an error.

        ######## Begin: arguments for `allennlp evaluate`##########
        subparser.add_argument(
            "--output-file",
            type=str,
            help="optional path to write the metrics to as JSON (for multiple "
            "files, put  between filenames e.g., output1.txt,output2.txt)",
        )

        subparser.add_argument(
            "--predictions-output-file",
            type=str,
            help="optional path to write the predictions to as JSON lines "
            "(for mutiple files, put  between filenames e.g., "
            "output1.jsonl,output2.jsonl)",
        )
        subparser.add_argument(
            "--weights-file",
            type=str,
            help="a path that overrides which weights file to use",
        )

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument(
            "--cuda-device", type=int, default=-1, help="id of GPU to use (if any)"
        )
        subparser.add_argument(
            "--batch-size",
            type=int,
            help="If non-empty, the batch size to use during evaluation.",
        )

        subparser.add_argument(
            "--batch-weight-key",
            type=str,
            default="",
            help="If non-empty, name of metric used to weight the loss on a per-batch basis.",
        )

        subparser.add_argument(
            "--extend-vocab",
            action="store_true",
            default=False,
            help="if specified, we will use the instances in your new dataset to "
            "extend your vocabulary. If pretrained-file was used to initialize "
            "embedding layers, you may also need to pass --embedding-sources-mapping.",
        )

        subparser.add_argument(
            "--embedding-sources-mapping",
            type=str,
            default="",
            help="a JSON dict defining mapping from embedding module path to embedding "
            "pretrained-file used during training. If not passed, and embedding needs to be "
            "extended, we will try to use the original file paths used during training. If "
            "they are not available we will use random vectors for embedding extension.",
        )
        subparser.add_argument(
            "--file-friendly-logging",
            action="store_true",
            default=False,
            help="outputs tqdm status on separate lines and slows tqdm refresh rate",
        )

        subparser.add_argument(
            "--include-package",
            type=str,
            action="append",
            default=[],
            help="additional packages to include",
        )
        subparser.add_argument(
            "--archive-file", type=str, help="path to an archived trained model"
        )
        subparser.add_argument(
            "--auto-names",
            default="NONE",
            help="Automatically create output names for each evaluation file. "
            "`NONE` will not automatically generate a file name for the "
            "neither the metrics nor the predictions. In this case you will"
            " need to pas in both `metrics_output_file` and `predictions_output_file`. "
            "`METRICS` will only automatically create a file name for the"
            " metrics file. `PREDS` will only automatically create a file"
            " name for the predictions outputs. `ALL` will create a "
            "filename for both the metrics and the predictions.",
            choices=["NONE", "METRICS", "PREDS", "ALL"],
        )
        ######## End: arguments for `allennlp train`##########

        ######## Begin: Specific keyword arguments for `allennlp evaluate_with_wandb`##########
        subparser.add_argument(
            "--update-run",
            action="store_true",
            help="Update existing run instead of creating a new one.",
        )
        subparser.add_argument(
            "--wandb-job-type",
            action=SetWandbEnvVar,
            type=str,
            default="eval",
        )
        ######## End: Specific keyword arguments for `allennlp evaluate_with_wandb`##########

        # we will not do anything if the subcommand is not train_with_wandb
        # because otherwise parse_known_args() can throw error or show train_with_wandb's help
        # even if we are asking for --help for some other command

        if sys.argv[1] != "evaluate-with-wandb":
            subparser.set_defaults(func=evaluate_from_args)

            return subparser
        # Add dynamic args for overrides and env variables
        known_args, hyperparams = subparser.parse_known_args(sys.argv[2:])
        all_args, hparams_for_overrides, env_vars = translate(hyperparams)
        overrides_json = f"--overrides={json.dumps(hparams_for_overrides)}"

        # update sys.argv with the json from
        sys.argv.append(overrides_json)
        # add all hyperparams in both froms--json as well as dot notation
        # we do this so that parser_args() in the allennlp code does not throw error

        for arg in all_args:
            subparser.add_argument(f"{arg}")

        # Add the rest of the arguments of `allennlp train` that we held out due to the feeler call to parse_known_args()
        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help=(
                "a json(net) structure used to override the experiment configuration, e.g., "
                "'{\"iterator.batch_size\": 16}'.  Nested parameters can be specified either"
                " with nested dictionaries or with dot syntax."
            ),
        )

        subparser.add_argument(
            "input_file",
            type=str,
            help="path to the file containing the evaluation data (for multiple "
            "files, put between filenames e.g., input1.txt,input2.txt)",
        )

        # set env vars
        os.environ.update(env_vars)

        subparser.set_defaults(func=evaluate_from_args)

        return subparser


def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return evaluate_from_archive(
        archive_file=args.archive_file,
        input_file=args.input_file,
        metrics_output_file=args.output_file,
        predictions_output_file=args.predictions_output_file,
        batch_size=args.batch_size,
        cmd_overrides=args.overrides,
        cuda_device=args.cuda_device,
        embedding_sources_mapping=args.embedding_sources_mapping,
        extend_vocab=args.extend_vocab,
        weights_file=args.weights_file,
        file_friendly_logging=args.file_friendly_logging,
        batch_weight_key=args.batch_weight_key,
        auto_names=args.auto_names,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_id=args.wandb_run_id,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        name=args.wandb_name,
        notes=args.wandb_notes,
        tags=args.wandb_tags,
        new_run=not args.update_run,
    )


def evaluate_from_archive(
    archive_file: Optional[Union[str, os.PathLike]],
    input_file: str,
    metrics_output_file: Optional[str] = None,
    predictions_output_file: Optional[str] = None,
    batch_size: Optional[int] = None,
    cmd_overrides: Union[str, Dict[str, Any]] = "",
    cuda_device: int = -1,
    embedding_sources_mapping: str = None,
    extend_vocab: bool = False,
    weights_file: str = None,
    file_friendly_logging: bool = False,
    batch_weight_key: str = None,
    auto_names: str = "NONE",
    project: Optional[str] = None,
    entity: Optional[str] = None,
    run_id: Optional[str] = None,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
    name: Optional[str] = None,
    notes: Optional[str] = None,
    tags: Optional[Union[str, List[str]]] = None,
    new_run: bool = True,
) -> Dict[str, Any]:
    """
    # Parameters
    archive_file: `Union[str, PathLike]`
        Path to an archived trained model.
    input_file: `str`
        path to the file containing the evaluation data (for multiple files,
         put ":" between filenames e.g., input1.txt:input2.txt)
    metrics_output_file: `str`, optional (default=`None`)
        optional path to write the metrics to as JSON (for multiple files, put
         ":" between filenames e.g., output1.txt:output2.txt)
    predictions_output_file: `str`, optional (default=`None`)
        "optional path to write the predictions to (for multiple files, put ":"
         between filenames e.g., output1.jsonl:output2.jsonl)
    batch_size: `int`, optional (default=`None`)
        If non-empty, the batch size to use during evaluation.
    cmd_overrides: `str`, optional (default=`""`)
        a json(net) structure used to override the experiment configuration,
         e.g., '{\"iterator.batch_size\": 16}'.  Nested parameters can be
          specified either with nested dictionaries or with dot syntax.
    cuda_device: `int`, optional (default=`-1`)
        id of GPU to use (if any)
    embedding_sources_mapping: `str`, optional (default=`None`)
        a JSON dict defining mapping from embedding module path to embedding
        pretrained-file used during training. If not passed, and embedding
        needs to be extended, we will try to use the original file paths used
        during training. If they are not available we will use random vectors
        for embedding extension.
    extend_vocab: `bool`, optional (default=`False`)
        if specified, we will use the instances in your new dataset to extend
        your vocabulary. If pretrained-file was used to initialize embedding
        layers, you may also need to pass --embedding-sources-mapping.
    weights_file:`str`, optional (default=`None`)
        A path that overrides which weights file to use
    file_friendly_logging : `bool`, optional (default=`False`)
        If `True`, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.
    batch_weight_key: `str`, optional (default=`None`)
        If non-empty, name of metric used to weight the loss on a per-batch basis.
    auto_names:`str`, optional (default=`"NONE"`)
        Automatically create output names for each evaluation file.`NONE` will
        not automatically generate a file name for the neither the metrics nor
        the predictions. In this case you will need to pas in both
        `metrics_output_file` and `predictions_output_file`. `METRICS` will only
        automatically create a file name for the metrics file. `PREDS` will only
        automatically create a file name for the predictions outputs. `ALL`
        will create a filename for both the metrics and the predictions.
    # Returns
    all_metrics: `Dict[str, Any]`
        The metrics from every evaluation file passed.
    """

    common_logging.FILE_FRIENDLY_LOGGING = file_friendly_logging

    # Disable some of the more verbose logging statements
    logging.getLogger("allennlp.common.params").disabled = True
    logging.getLogger("allennlp.nn.initializers").disabled = True
    logging.getLogger("allennlp.modules.token_embedders.embedding").setLevel(
        logging.INFO
    )

    # wandb args
    logger.debug("Wandb related varaibles")
    logger.debug(
        "%s |   %s  |   %s",
        "variable".ljust(15),
        "value from env".ljust(50),
        "value in constructor".ljust(50),
    )

    for e, a in [("PROJECT", project), ("ENTITY", entity)]:
        logger.debug(
            "%s |   %s  |   %s",
            str(e).lower()[:15].ljust(15),
            str(read_from_env("WANDB_" + e))[:50].ljust(50),
            str(a)[:50].ljust(50),
        )
    logger.debug("All wandb related envirnment varaibles")
    logger.debug("%s |   %s  ", "ENV VAR.".ljust(15), "VALUE".ljust(50))

    for k, v in os.environ.items():
        if "WANDB" in k or "ALLENNLP" in k:
            logger.debug(
                "%s |   %s  ",
                str(k)[:15].ljust(15),
                str(v)[:50].ljust(50),
            )
    t = read_from_env("WANDB_TAGS") or tags

    if isinstance(t, str):
        tags = t.split(",")
    else:
        tags = t
    if tags is not None:
        tags.append("job-type@eval-only")
    else:
        tags = ["job-type@eval-only"]

    if run_id is not None:
        tags.append(f"reference-for-eval@{entity}/{project}/{run_id}")

    import wandb

    if archive_file is None:
        assert run_id is not None
        with tempfile.TemporaryDirectory() as download_dir:
            api = wandb.Api()
            ref_run = api.run(
                f"{entity or read_from_env('WANDB_ENTITY')}/{project or read_from_env('WANDB_PROJECT')}/{run_id}"
            )
            ref_model_file = None
            if tags:
                tags += ref_run.tags
            for file in ref_run.files():
                if pathlib.PurePath(file.name).match(MODEL_FILE):
                    ref_model_file = file
        if ref_model_file is None:
            raise ValueError(f"{ref_run} does not have {MODEL_FILE}")
        logger.info(f"Downloading model file to {download_dir}")
        ref_model_file.download(download_dir)
        # Load from archive
        archive = load_archive(
            Path(download_dir) / ref_model_file.name,
            weights_file=weights_file,
            cuda_device=cuda_device,
            overrides=cmd_overrides,
        )
    else:
        # Load from archive
        archive = load_archive(
            archive_file,
            weights_file=weights_file,
            cuda_device=cuda_device,
            overrides=cmd_overrides,
        )

    if new_run:
        new_run_id = wandb.util.generate_id()
        wandb_resume = "never"
    else:
        if run_id is None:
            raise ValueError(f"new_run is False. Provide run.")
        else:
            new_run_id = run_id
            wandb_resume = "must"

    wandb_run = wandb.init(
        id=new_run_id,
        project=project or read_from_env("WANDB_PROJECT"),
        entity=entity or read_from_env("WANDB_ENTITY"),
        group=group or read_from_env("WANDB_GROUP"),
        name=name or read_from_env("WANDB_NAME"),
        notes=notes or read_from_env("WANDB_NOTES"),
        job_type=job_type or read_from_env("WANDB_JOB_TYPE"),
        tags=tags,
    )

    config = deepcopy(archive.config)
    wandb_run.config.update(config.as_dict())
    prepare_environment(config)
    model = archive.model
    model.eval()

    # Load the evaluator from the config key `Evaluator`
    evaluator_params = config.pop("evaluation", {})
    evaluator_params["cuda_device"] = cuda_device
    evaluator = Evaluator.from_params(evaluator_params)

    # Load the evaluation data
    dataset_reader = archive.validation_dataset_reader

    # split files
    try:
        # Try reading it as a list of JSON objects first. Some readers require
        # that kind of input.
        evaluation_data_path_list = json.loads(f"[{input_file}]")
    except JSONDecodeError:
        evaluation_data_path_list = input_file.split(",")

    # TODO(gabeorlanski): Is it safe to always default to .outputs and .preds?
    # TODO(gabeorlanski): Add in way to save to specific output directory
    if metrics_output_file is not None:
        if auto_names == "METRICS" or auto_names == "ALL":
            logger.warning(
                f"Passed output_files will be ignored, auto_names is set to {auto_names}"
            )

            # Keep the path of the parent otherwise it will write to the CWD
            assert all(isinstance(p, str) for p in evaluation_data_path_list), (
                "When specifying JSON blobs as input, the output files must be explicitly named with "
                "--output-file."
            )
            output_file_list = [
                p.parent.joinpath(f"{p.stem}.outputs")
                for p in map(Path, evaluation_data_path_list)
            ]
        else:
            output_file_list = metrics_output_file.split(",")  # type: ignore
            assert len(output_file_list) == len(evaluation_data_path_list), (
                "The number of `metrics_output_file` paths must be equal to the number "
                "of datasets being evaluated."
            )
    if predictions_output_file is not None:
        if auto_names == "PREDS" or auto_names == "ALL":
            logger.warning(
                f"Passed predictions files will be ignored, auto_names is"
                f" set to {auto_names}"
            )

            # Keep the path of the parent otherwise it will write to the CWD
            assert all(isinstance(p, str) for p in evaluation_data_path_list), (
                "When specifying JSON blobs as input, the predictions output files must be explicitly named with "
                "--predictions-output-file."
            )
            predictions_output_file_list = [
                p.parent.joinpath(f"{p.stem}.preds")
                for p in map(Path, evaluation_data_path_list)
            ]
        else:
            predictions_output_file_list = predictions_output_file.split(",")  # type: ignore
            assert len(predictions_output_file_list) == len(
                evaluation_data_path_list
            ), (
                "The number of `predictions_output_file` paths must be equal"
                + "to the number of datasets being evaluated. "
            )

    # output file
    output_file_path = None
    predictions_output_file_path = None

    # embedding sources
    if extend_vocab:
        logger.info("Vocabulary is being extended with embedding sources.")
        embedding_sources = (
            json.loads(embedding_sources_mapping) if embedding_sources_mapping else {}
        )

    all_metrics = {}
    for index, evaluation_data_path in enumerate(evaluation_data_path_list):
        config = deepcopy(archive.config)

        # Get the eval file name so we can save each metric by file name in the
        # output dictionary.
        if isinstance(evaluation_data_path, str):
            eval_file_name = Path(evaluation_data_path).stem
        else:
            eval_file_name = str(index)

        if metrics_output_file is not None:
            # noinspection PyUnboundLocalVariable
            output_file_path = output_file_list[index]

        if predictions_output_file is not None:
            # noinspection PyUnboundLocalVariable
            predictions_output_file_path = predictions_output_file_list[index]

        logger.info("Reading evaluation data from %s", eval_file_name)
        data_loader_params = config.get("validation_data_loader", None)
        if data_loader_params is None:
            data_loader_params = config.get("data_loader")
        if batch_size:
            data_loader_params["batch_size"] = batch_size
        data_loader = DataLoader.from_params(
            params=data_loader_params,
            reader=dataset_reader,
            data_path=evaluation_data_path,
        )

        if extend_vocab:
            logger.info("Vocabulary is being extended with test instances.")
            model.vocab.extend_from_instances(instances=data_loader.iter_instances())
            # noinspection PyUnboundLocalVariable
            model.extend_embedder_vocab(embedding_sources)

        data_loader.index_with(model.vocab)

        metrics = evaluator(
            model,
            data_loader,
            batch_weight_key,
            metrics_output_file=output_file_path,
            predictions_output_file=predictions_output_file_path,
        )

        # Add the metric prefixed by the file it came from.
        for name, value in metrics.items():
            if len(evaluation_data_path_list) > 1:
                key = f"{eval_file_name}_"
            else:
                key = ""
            all_metrics[f"{key}{name}"] = value

    logger.info("Finished evaluating.")

    _metrics = {}
    for key, value in all_metrics.items():
        _metrics["test_" + key] = value

    logger.info("Updating summary on wandb.")
    wandb.log(_metrics)
    wandb.summary.update(_metrics)
    wandb.finish(0)
    return all_metrics
