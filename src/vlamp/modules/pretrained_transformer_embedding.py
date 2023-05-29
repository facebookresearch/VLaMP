# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from typing import Dict, List, Optional, Union
from allennlp.modules.transformer.layer_norm import LayerNorm
from .pretrained_transformer_module import PretrainedTransformerModule
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class PretrainedTransformerTokenEmbeddings(PretrainedTransformerModule):
    _pretrained_allow_missing: Optional[List[str]] = [r"extension_weight"]

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        pad_token_id: int = None,
        output_size: Optional[int] = None,
        num_extensions: int = 0,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((vocab_size, embedding_size)))
        self.pad_token_id = pad_token_id
        if num_extensions > 0:
            self.extension_weight = torch.nn.Parameter(
                torch.empty((num_extensions, embedding_size))
            )
        # self.word_embeddings = torch.nn.Embedding(
        #    vocab_size, embedding_size, padding_idx=pad_token_id
        # )
        self.linear_transform_or_identity: Union[torch.nn.Linear, torch.nn.Identity] = (
            torch.nn.Linear(embedding_size, output_size)
            if output_size
            else torch.nn.Identity()
        )
        self.reset_parameters()

    def get_weight(self) -> torch.Tensor:
        if hasattr(self, "extension_weight"):
            return torch.cat([self.weight, self.extension_weight], dim=0)
        else:
            return self.weight

    def init_extension(self) -> None:
        with torch.no_grad():
            logger.info(
                "Initializing embeddings extension using pre-trained embeddings."
            )
            self.extension_weight.copy_(self.weight.mean(dim=0).unsqueeze(0))

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight)
        if hasattr(self, "extension_weight"):
            torch.nn.init.normal_(self.extension_weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.pad_token_id is not None:
            with torch.no_grad():
                self.weight[self.pad_token_id].fill_(0)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.linear_transform_or_identity(
            F.embedding(ids, self.get_weight(), padding_idx=self.pad_token_id)
        )

    def get_output_dim(self) -> int:
        if not isinstance(self.linear_transform_or_identity, torch.nn.Identity):
            return self.linear_transform_or_identity.out_features
        else:
            return self.weight.shape[-1]

    def load_state_dict(self, state_dict, strict: bool = True):
        value = super().load_state_dict(state_dict, strict)
        if "extension_weight" not in state_dict:
            self.init_extension()
        return value

    # @classmethod
    # def add_to_kwargs_for_from_config(
    #    cls, config: "PretrainedConfig", **kwargs
    # ) -> Dict:
    #    kwarg_dict = {"num_extension": kwargs.get("num_extension", 0)}
    #    return kwarg_dict


class PretrainedTransformerPositionEmbeddings(PretrainedTransformerModule):
    def __init__(
        self,
        position_vocab_size: int,
        type_vocab_size: int,
        embedding_size: int,
        position_pad_token_id: int = 0,
        output_size: Optional[int] = None,
        layer_norm_eps: Optional[float] = 1e-12,
        dropout: float = 0.1,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        if position_vocab_size > 0:
            self.position_embeddings = torch.nn.Embedding(
                position_vocab_size,
                embedding_size,
                padding_idx=position_pad_token_id,
            )
        if type_vocab_size > 0:
            self.type_embeddings = torch.nn.Embedding(type_vocab_size, embedding_size)
        if output_size:
            self.linear_transform = torch.nn.Linear(embedding_size, output_size)
        if use_layer_norm:
            self.layer_norm = LayerNorm(embedding_size, eps=layer_norm_eps)
        else:
            self.layer_norm = torch.nn.Identity()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(  # type: ignore
        self,
        token_reps: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        reps = [token_reps] if token_reps is not None else []
        if hasattr(self, "position_embeddings") and position_ids is not None:
            reps.append(self.position_embeddings(position_ids))

        if hasattr(self, "type_embeddings") and token_type_ids is not None:
            reps.append(self.type_embeddings(token_type_ids))

        # Modified super's forward
        outputs = sum(reps)  # type: ignore
        outputs = self.layer_norm(outputs)
        outputs = self.dropout(outputs)
        embeddings = outputs

        if hasattr(self, "linear_transform"):
            embeddings = self.linear_transform(embeddings)
        assert isinstance(embeddings, torch.Tensor)
        return embeddings
