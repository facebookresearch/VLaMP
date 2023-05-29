# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from collections import OrderedDict
from typing import List, Optional, Union, Dict, Type, Any, TypeVar
from allennlp.common import Registrable
from allennlp.common.testing import TransformerModule
from allennlp.modules.transformer import (
    TransformerEmbeddings,
)
from os import PathLike
import torch.distributed as dist
from allennlp.common.util import is_distributed
from allennlp.nn.util import (
    _check_incompatible_keys,
)
import torch
from transformers import AutoConfig, PreTrainedModel
import re
import logging

logger = logging.getLogger(__name__)


_T = TypeVar("_T", bound="TransformerModule")


def process_ignore_on_state_dict(
    state: Dict[str, torch.Tensor], ignore: Optional[List[str]] = None
):
    """Based on https://github.com/allenai/allennlp/blob/v2.9.3/allennlp/nn/util.py#L930"""
    out: Dict[str, torch.Tensor] = OrderedDict()
    ignore_used: Optional[List[bool]] = [False] * len(ignore) if ignore else None
    for key in state.keys():
        ignore_key = False
        if ignore:
            for i, pattern in enumerate(ignore):
                if re.match(pattern, key):
                    if ignore_used:
                        ignore_used[i] = True
                    logger.warning("ignoring %s from state dict", key)
                    ignore_key = True
                    break

        if ignore_key:
            continue

        new_key = key
        out[new_key] = state[key]
    return out


class PretrainedTransformerModule(TransformerModule, Registrable):
    @classmethod
    def from_pretrained_module(
        cls: Type[_T],
        model_name: str,
        *,
        load_weights: bool = True,
        weights_path: Optional[Union[str, PathLike]] = None,
        pretrained_model: Optional[PreTrainedModel] = None,
        auto_config_kwargs: Optional[Dict[str, Any]] = None,
        mapping: Optional[Dict[str, str]] = None,
        relevant_module: Optional[Union[str, List[str]]] = None,
        ignore: Optional[List[str]] = None,
        allow_missing: Optional[List[str]] = None,
        strict: bool = True,
        **kwargs,
    ) -> _T:
        if pretrained_model is None:
            return super().from_pretrained_module(
                model_name,
                load_weights=load_weights,
                weights_path=weights_path,
                auto_config_kwargs=auto_config_kwargs,
                mapping=mapping,
                relevant_module=relevant_module,
                ignore=ignore,
                allow_missing=allow_missing,
                strict=strict,
                **kwargs,
            )
        else:
            config = getattr(
                pretrained_model,
                "config",
                AutoConfig.from_pretrained(model_name, **(auto_config_kwargs or {})),
            )
            model = cls._from_config(config, **kwargs)
            if load_weights:
                state = process_ignore_on_state_dict(
                    pretrained_model.state_dict(),
                    ignore=ignore if ignore is not None else cls._pretrained_ignore,
                )
                state_dict = model._get_mapped_state_dict(state, mapping=mapping)
                logger.info("Loading state_dict into module")

                if not is_distributed():
                    assert state_dict is not None
                    missing_keys, unexpected_keys = model.load_state_dict(
                        state_dict, strict=False
                    )
                else:
                    # We're in distributed training. `state_dict` is `None` for all process groups
                    # except the global primary.
                    # Syncronize here since non-primary process groups will have to wait for the primary
                    # to load the state_dict into memory.
                    dist.barrier()
                    # Now load the state dict into the model.
                    missing_keys, unexpected_keys = model.load_state_dict_distributed(
                        state_dict, strict=False
                    )

                # Exclude any keys in `missing_keys` that match with the `allow_missing`
                # regular expressions.
                if allow_missing is None:
                    allow_missing = cls._pretrained_allow_missing
                if allow_missing:
                    missing_keys = [
                        k
                        for k in missing_keys
                        if not any(re.match(p, k) for p in allow_missing)
                    ]
                _check_incompatible_keys(model, missing_keys, unexpected_keys, strict)
            return model
