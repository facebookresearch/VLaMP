# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from typing import Optional, Union
from allennlp.modules.transformer import TransformerStack
from allennlp.modules.transformer.transformer_stack import TransformerStackOutput
import torch
from .frame_encoders import ObservationFeaturesEncoder, FeatureContextualizer


@FeatureContextualizer.register("transformer")
class TransformerFeatureContextualizer(FeatureContextualizer):
    def __init__(
        self,
        input_size: int,
        num_hidden_layers: int = 4,
        intermediate_size: Optional[int] = None,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        activation: Union[str, torch.nn.Module] = "relu",
        num_cross_attention_tokens: Optional[int] = None,
        output_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            input_size, output_size=input_size if output_size is None else output_size
        )
        if intermediate_size is None:
            intermediate_size = 2 * input_size
        self.transformer = TransformerStack(
            num_hidden_layers=num_hidden_layers,
            hidden_size=input_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            activation=activation,
            add_cross_attention=(num_cross_attention_tokens is not None),
        )
        if num_cross_attention_tokens:
            self.cross_attention_emb = torch.nn.Parameter(
                torch.empty(num_cross_attention_tokens, input_size)
            )
            torch.nn.init.normal_(self.cross_attention_emb)
        else:
            self.cross_attention_emb = None

        if output_size:
            self.output_projector = torch.nn.Linear(input_size, output_size)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        bs, seq, num_toks, feature_size = inp.shape
        # inp = inp.view(bs, -1, feature_size)
        inp = inp.view(bs * seq, num_toks, feature_size)
        for w in self.transformer.parameters():
            dtype = w.dtype
            break
        inp = inp.to(dtype=dtype)
        if self.cross_attention_emb is not None:
            output = self.transformer(
                inp, encoder_hidden_states=self.cross_attention_emb.unsqueeze(0)
            ).final_hidden_states
            output = output.view(
                bs, seq, self.cross_attention_emb.shape[0], feature_size
            )
        else:
            output = self.transformer(inp).final_hidden_states
            output = output.view(bs, seq, num_toks, feature_size)

        if hasattr(self, "output_projector"):
            output = self.output_projector(output)

        return output


@FeatureContextualizer.register("fixed-length-transformer")
class FixedLengthTransformerFeatureContextualizer(FeatureContextualizer):
    def __init__(
        self,
        input_size: int,
        num_tokens: int,
        num_hidden_layers: int = 4,
        intermediate_size: Optional[int] = None,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        activation: Union[str, torch.nn.Module] = "relu",
        output_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            input_size, output_size=input_size if output_size is None else output_size
        )
        if intermediate_size is None:
            intermediate_size = 2 * input_size
        self.transformer = TransformerStack(
            num_hidden_layers=num_hidden_layers,
            hidden_size=input_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            activation=activation,
            add_cross_attention=True,
        )
        self.num_tokens = num_tokens
        self.fixed_embeddings = torch.nn.Parameter(torch.empty(num_tokens, input_size))
        torch.nn.init.normal_(self.fixed_embeddings)

        if output_size:
            self.output_projector = torch.nn.Linear(input_size, output_size)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        assert len(inp.shape) == 4
        bs, seq, num_toks, feature_size = inp.shape
        inp = inp.view(bs * seq, num_toks, feature_size)
        with torch.no_grad():
            mask = ~(inp.sum(-1).sum(-1) == 0)
        output = self.transformer(
            self.fixed_embeddings.unsqueeze(0).expand(
                bs * seq, self.num_tokens, feature_size
            ),
            encoder_hidden_states=inp,
            encoder_attention_mask=mask,
        ).final_hidden_states
        output = output.view(bs, seq, self.num_tokens, feature_size)

        if hasattr(self, "output_projector"):
            output = self.output_projector(output)

        return output
