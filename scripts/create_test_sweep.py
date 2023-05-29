# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from typing import Tuple, List, Dict
from wandb_allennlp.commands.parser_base import WandbParserBase
from allennlp.commands import Subcommand
import jsonlines
from allennlp.data.vocabulary import Vocabulary
import argparse
from pathlib import Path
import os
from matplotlib import pyplot as plt
import wandb
import shutil
from copy import deepcopy
import logging
import tqdm
from wandb_utils.commands.best_models import find_best_model
from wandb_utils import find_best_models_in_sweeps, get_config_file_for_run
import pandas as pd
from jinja2 import Environment, Template, meta
import json
import yaml
import re

logger = logging.getLogger(__name__)


def add_target_config_path(args: argparse.Namespace, variables: Dict) -> Dict:
    if args.target_config_path is not None:
        target_path = Path(args.target_config_path)
    elif args.target_config_dir is not None:
        if "sweep_name" in variables:
            target_path = (
                Path(args.target_config_dir) / variables["sweep_name"]
            ).with_suffix(".jsonnet")
        elif "tags" in variables:
            values: List[str] = []

            for tag in variables["tags"].split("|"):
                values.append(tag.split("@")[-1].strip())
            target_path = (Path(args.target_config_dir) / "-".join(values)).with_suffix(
                ".jsonnet"
            )
        else:
            raise ValueError("Provide --target_config_path")

    if target_path.exists() and not args.overwrite_config_file:
        raise ValueError(
            f"{target_path} already exists. Set --overwrite-config-file if you want to overwrite"
        )

    variables["target_model_config"] = str(target_path)

    return variables


def add_target_sweep_name(args: argparse.Namespace, variables: Dict) -> Dict:
    if args.sweep_name_pattern:
        target_sweep_name = args.sweep_name_pattern.format_map(
            {
                "sweep_name_prefix": (args.sweep_name_prefix or "best"),
                "sweep_name_suffix": (args.sweep_name_suffix or "test"),
                **variables,
            }
        )
    else:
        if "sweep_name" in variables:
            target_sweep_name = "-".join(
                [
                    args.sweep_name_prefix,
                    variables["sweep_name"],
                    args.sweep_name_suffix,
                ]
            )
        elif "tags" in variables:
            values: List[str] = []

            for tag in variables["tags"].split("|"):
                values.append(tag.split("@")[-1].strip())
            target_sweep_name = "-".join(
                [args.sweep_name_prefix] + values + [args.sweep_name_suffix]
            )
    variables["target_sweep_name"] = target_sweep_name

    return variables


def update_training_data_path(args: argparse.Namespace, variables: Dict) -> Dict:

    variables["target_train_data_path"] = (
        "'\"{"
        + variables["train_data_path"]
        + ","
        + variables["validation_data_path"]
        + "}\"'"
    )

    return variables


def add_variables(args: argparse.Namespace, variables: Dict) -> Dict:
    # Append to tags
    variables["target_tags"] = ",".join(
        variables.get("tags", "previous-tags@unk").split("|")
        + [
            "run-type@test",
            f"reference@{args.wandb_entity}/{args.wandb_project}/{variables['run']}",
        ]
        + list(args.wandb_tags.split(","))
    )

    # target config file path
    logger.debug("Adding target config path")
    add_target_config_path(args, variables)

    #
    variables["tracked_metric"] = "test_fixed_f1"

    #
    add_target_sweep_name(args, variables)

    # Combine train and validation data
    # update_training_data_path(args, variables)

    return variables


def main(args: argparse.Namespace) -> None:
    if args.wandb_run_id is None:
        if args.wandb_sweep_id is None:
            raise ValueError("Set either --wandb-run-id or --wandb-sweep-id")

        if args.metric is None:
            raise ValueError("Set either --wandb-run-id or --metric")

        if args.metric[0] not in ["-", "+"]:
            raise ValueError(
                "--metric should be prepended with + (for higher the better) or - (lower the better)."
                " For example, +accuracy or -error"
            )

    if args.wandb_target_project is None:
        args.wandb_target_project = args.wandb_project

    api = wandb.Api()  # type: ignore

    if args.wandb_run_id is not None:
        run = api.run(f"{args.wandb_entity}/{args.wandb_project}/{args.wandb_run_id}")
        # Taken from https://github.com/dhruvdcoder/wandb-utils/src/wandb_utils/misc.py#L115
        # TODO: At some point move this logic into wandb-utils lib
        summary_list = []
        config_list = []
        name_list = []
        sweep_list = []

        # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files
        summary_list.append({k: v for k, v in run.summary._json_dict.items()})

        # run.config is the input metrics.  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # run.name is the name of the run.
        name_list.append(
            {
                "run": run.id,
                "run_name": run.name,
                "entity": run.entity,
                "project": run.project,
                "path": f"{run.entity}/{run.project}/{run.id}",
                "tags": "|".join(run.tags),
            }
        )

        # sweep
        sweep_list.append(
            {
                "sweep": run.sweep.id,
                "sweep_name": run.sweep.config.get("name", ""),
            }
            if run.sweep
            else {"sweep": "", "sweep_name": ""}
        )

        summary_df = pd.DataFrame.from_records(summary_list)
        config_df = pd.DataFrame.from_records(config_list)
        sweep_df = pd.DataFrame.from_records(sweep_list)
        name_df = pd.DataFrame.from_records(name_list)
        df = pd.concat([name_df, sweep_df, config_df, summary_df], axis=1)

    else:
        metric = args.metric[1:].strip()
        maximum = args.metric[0] == "+"
        df = find_best_models_in_sweeps(
            entity=args.wandb_entity,
            project=args.wandb_project,
            sweep=args.wandb_sweep_id,
            metric=metric,
            maximum=maximum,
            api=api,
        )
    assert len(df) == 1
    run_data = df.iloc[0].to_dict()
    variables = add_variables(args, run_data)

    # read the yaml template
    with open(args.sweep_template) as f:
        sweep_template_str = f.read()

    jinja_env = Environment(extensions=["jinja2.ext.do"])
    sweep_template = jinja_env.from_string(sweep_template_str)
    ast = jinja_env.parse(sweep_template_str)
    sbatch_jinja_variables = meta.find_undeclared_variables(ast)  # type: ignore
    sweep_config_str = sweep_template.render(**variables)

    # get config file for the run
    get_config_file_for_run(
        entity=args.wandb_entity,
        project=args.wandb_project,
        run_id=variables["run"],
        relative_path="config.json",
        output_path=variables["target_model_config"],
        api=api,
    )
    # Add seed sharing
    # See ref: https://github.com/dhruvdcoder/wandb-utils/blob/main/src/wandb_utils/misc.py
    with open(variables["target_model_config"]) as json_f:
        logger.info("Adding tied seed variable to model config.")
        model_config = json.load(json_f)
        # Dhruvesh: This is a hack to add model_archives
        callbacks = model_config["trainer"].get("callbacks")
        if callbacks:
            for cb in callbacks:
                if isinstance(cb, dict):
                    if cb.get("type") == "wandb_allennlp":
                        cb["save_model_archive"] = True
            callbacks.append('checkpoint_to_best')
        # introduce seed as an env variable
        placeholder = "__temp__seed__"
        seed_parameters = ["random_seed", "pytorch_seed", "numpy_seed"]

        for param in seed_parameters:
            model_config[param] = placeholder
        # Make cuda_device configurable
        model_config["trainer"]["cuda_device"] = "cuda_device"
        model_config_str = (
            "local seed = std.parseJson(std.extVar('seed'));\n"
            + "local cuda_device = std.parseJson(std.extVar('CUDA_DEVICE'));\n"
            + json.dumps(model_config, sort_keys=True, indent=2)
        )
        # Now replace the placeholder with the variable
        regex = r"\"__temp__seed__\""
        model_config_str = re.sub(
            regex,
            "seed",
            model_config_str,
            len(seed_parameters),
            re.MULTILINE,
        )

    with open(variables["target_model_config"], "w") as f:
        f.write(model_config_str)

    if not args.skip_create_sweep:
        # create sweep
        sweep_id = wandb.sweep(  # type:ignore
            yaml.safe_load(sweep_config_str),
            entity=args.wandb_entity,
            project=args.wandb_target_project,
        )
        logger.info(
            f"Created sweep {sweep_id} in {args.wandb_entity}/{args.wandb_target_project} with following config."
        )
        logger.info(f"\n{sweep_config_str}")


@Subcommand.register("create-test-sweep")
class create_test_sweep(WandbParserBase):
    description = (
        "Create a sweep with multiple random sweeds using a run/sweep as reference"
    )
    help_message = "Use this subcommand to create a test sweep which starts 10 runs using different random seeds"
    require_run_id = False
    entry_point = main

    def add_arguments(
        self, subparser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        subparser.add_argument(
            "-t",
            "--sweep-template",
            type=Path,
            required=True,
            help="Path to jinja2 template for the yaml sweep config. "
            "You can use any variables in the run config of the wandb run",
        )
        subparser.add_argument(
            "--target-config-dir",
            type=Path,
            default=Path("best_models_configs2"),
            help="Path to dump the retrieved json/jsonnet file.",
        )
        subparser.add_argument(
            "--target-config-path",
            type=Path,
            help="Path to dump the retrieved json/jsonnet file.",
        )
        subparser.add_argument(
            "--overwrite-config-file",
            action="store_true",
            help="Overwrite target config path if file exists",
        )
        subparser.add_argument(
            "--skip-create-sweep",
            action="store_true",
            help="Set to not create a new sweep.",
        )
        subparser.add_argument(
            "--sweep-name-prefix",
            type=str,
            default="best",
            help="Prefix for the sweep name. Ignored when --sweep-name is provided without patterns. (default: best)",
        )
        subparser.add_argument(
            "--sweep-name-suffix",
            type=str,
            default="test",
            help="Suffix for the sweep name. Ignored when --sweep-name is provided without patterns. (default: '')",
        )
        subparser.add_argument(
            "--sweep-name-pattern",
            type=str,
            default="{sweep_name_prefix}-{sweep_name}-{sweep_name_suffix}",
            help="Complete sweep name or pattern for the target sweep.",
        )
        subparser.add_argument(
            "-s",
            "--wandb-sweep-id",
            type=str,
            help="Sweep from which the best model is picked (needed when --wandb-run-id is not provided)",
        )
        subparser.add_argument(
            "-m",
            "--metric",
            type=str,
            help="Name of the metric, prepended with +/-, to pick the best model from a sweep. Ex: +accuracy, -prediction_error, etc.",
        )
        subparser.add_argument(
            "-m",
            "--wandb-target-project",
            type=str,
            help="Project in which to create the sweep. If not provided --wandb-project will be used.",
        )

        subparser.set_defaults(func=main)

        return subparser
