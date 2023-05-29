# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from collections import OrderedDict
from typing import List, Optional, Union, Dict, Type, Any, TypeVar

from .pretrained_transformer_embedding import (
    PretrainedTransformerTokenEmbeddings,
    PretrainedTransformerPositionEmbeddings,
)
import logging

logger = logging.getLogger(__name__)


@PretrainedTransformerTokenEmbeddings.register("gpt2", "from_pretrained_module")
class GPT2TokenEmbeddings(PretrainedTransformerTokenEmbeddings):
    _pretrained_relevant_module: Optional[Union[str, List[str]]] = None
    _pretrained_mapping: Dict[str, str] = {
        "wte.weight": "weight",
    }
    _pretrained_ignore: Optional[List[str]] = [r"^h.*", r"^ln_.*", r"^wpe.*"]

    @classmethod
    def _from_config(cls, config: "PretrainedConfig", **kwargs):
        final_kwargs = {
            "vocab_size": config.vocab_size,
            "embedding_size": config.n_embd,
        }

        final_kwargs.update(**kwargs)
        instance = cls(**final_kwargs)
        # remove layer norm as GPT2 does not use it after word embeddings
        return instance


class GPT2PositionEmbeddings(PretrainedTransformerPositionEmbeddings):
    _pretrained_relevant_module: Optional[Union[str, List[str]]] = None
    _pretrained_mapping: Dict[str, str] = {
        "wpe": "position_embeddings",
    }
    _pretrained_ignore: Optional[List[str]] = [r"^h.*", r"^ln_.*", f"^wte.*"]

    @classmethod
    def _from_config(cls, config: "PretrainedConfig", **kwargs):
        final_kwargs = {}
        final_kwargs.update(
            {
                "embedding_size": config.n_embd,
                "position_vocab_size": config.n_positions,
                "type_vocab_size": 0,  # this is what the default in HF code is. I am not sure.
                "layer_norm_eps": config.layer_norm_epsilon,
                "use_layer_norm": False,
            }
        )

        final_kwargs.update(**kwargs)
        instance = cls(**final_kwargs)
        # remove layer norm as GPT2 does not use it after word embeddings
        return instance


PretrainedTransformerTokenEmbeddings.register("gpt2-large", "from_pretrained_module")(
    GPT2TokenEmbeddings
)
