# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from typing import Dict, Optional, Tuple
from vlamp.dataset_readers.open_domain_procedural_planning import (
    ExtendableTokenizer,
)
import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from vlamp.modules.pretrained_transformer_module import (
    PretrainedTransformerModule,
)


class PretrainedTransfomerDecoder(PretrainedTransformerModule):
    def __init__(self, pretrained_model: PreTrainedModel) -> None:
        super().__init__()
        self.pretrained_model = pretrained_model

    def init_decoder_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Dict:
        pass

    def get_output_dim(self) -> int:
        pass

    def num_attention_layers(self) -> int:
        # ISSUE: Hard coded for GPT2
        return len(self.pretrained_model.h)

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        past_key_values: Optional[
            Tuple[Tuple[torch.Tensor]]
        ] = None,  # needed to provide history. Also dictates the shape of output.
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[
            bool
        ] = False,  # need to make the hf transformer output history.
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        # As with out general decoding setup, in as noted in decoder.py
        # We assume previous_* contains all history including the current input
        outputs = self.pretrained_model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            use_cache=use_cache,
            return_dict=True,
        )
        return outputs


if __name__ == "__main__":
    from transformers import AutoModel, AutoTokenizer
    from vlamp.modules.gpt2_embedding import GPT2TokenEmbeddings

    model = AutoModel.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    emb = GPT2TokenEmbeddings.from_pretrained_module("gpt2", pretrained_model=model)
    decoder = PretrainedTransfomerDecoder(pretrained_model=model)
    inputs = tokenizer("Hello my friend")
    token_embs = emb(inputs["token_ids"])
    decoder(embs)
