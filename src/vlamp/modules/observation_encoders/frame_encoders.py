# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from operator import mod
from typing import Optional, Type
from allennlp.common import FromParams, Registrable
from allennlp.common.cached_transformers import AutoModel
from allennlp.modules import FeedForward
import torch
import torch.nn.functional as F
from transformers import CLIPVisionModel


class FeatureProjector(torch.nn.Module, Registrable):
    default_implementation: Optional[str] = "identity"

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        return self.output_size

    def get_input_dim(self) -> int:
        return self.input_size


@FeatureProjector.register("identity")
class IdentityFeatureProjector(FeatureProjector):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__(input_size, output_size)
        assert self.input_size == self.output_size

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return inp


@FeatureProjector.register("feedforward")
class FeedForwardFeatureProjector(FeatureProjector):
    def __init__(self, feedforward: FeedForward) -> None:
        super().__init__(feedforward.get_input_dim(), feedforward.get_output_dim())
        self.feedforward = feedforward

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.feedforward(inp)


@FeatureProjector.register("clip")
class CLIPFeatureProjector(FeatureProjector):
    def __init__(
        self,
        input_size: int = 224,
        output_size: int = 768,
        model_name: str = "openai/clip-vit-base-patch32",
    ) -> None:
        super().__init__(input_size, output_size)
        self.model_name = model_name
        self.projector = CLIPVisionModel.from_pretrained(model_name)
        self.projector.requires_grad_(False)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        assert len(inp.shape) == 5  # (bs, seq, C, H, W)
        bs, seq, C, H, W = inp.shape
        # There could be pads in the seq dimension
        with torch.no_grad():
            mask = ~(inp.sum(-1).sum(-1).sum(-1) == 0)
        result = self.projector(
            inp.reshape(bs * seq, C, H, W)
        ).last_hidden_state  # (bs*seq, num_clip_tokens, clip_vision_hidden_size)
        trailing_dims = result.shape[1:]
        result = result.reshape(bs, seq, *trailing_dims)
        result = result * mask.reshape(bs, seq).unsqueeze(-1).unsqueeze(-1)
        return result


class FeatureContextualizer(torch.nn.Module, Registrable):
    default_implementation: Optional[str] = "mean"

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    def get_output_dim(self) -> int:
        return self.output_size

    def get_input_dim(self) -> int:
        return self.output_size

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        assert self.input_size == self.output_size
        return inp


FeatureContextualizer.register("identity")(FeatureContextualizer)


@FeatureContextualizer.register("mean")
class MeanFeatureContextualizer(FeatureContextualizer):
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        assert self.get_input_dim() == self.get_output_dim()
        return torch.mean(inp, dim=-2)


class ObservationFeaturesEncoder(torch.nn.Module, Registrable):
    default_implementation: Optional[str] = "mean"

    def __init__(
        self,
        projector: FeatureProjector,
        contextualizer: FeatureContextualizer,
        reverse: bool = False,
    ) -> None:
        super().__init__()
        self.projector = projector
        self.contextualizer = contextualizer
        self.reverse = reverse

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.reverse:
            return self.projector(self.contextualizer(features))
        else:
            return self.contextualizer(self.projector(features))

    def get_input_dim(self) -> int:
        if self.reverse:
            return self.contextualizer.get_input_dim()
        else:
            return self.projector.get_input_dim()

    def get_output_dim(self) -> int:
        if self.reverse:
            return self.projector.get_output_dim()
        else:
            return self.contextualizer.get_output_dim()

    @classmethod
    def construct_mean_feature_encoder(
        cls: Type["ObservationFeaturesEncoder"], feature_size: int
    ) -> "ObservationFeaturesEncoder":
        return cls(
            projector=IdentityFeatureProjector(
                input_size=feature_size, output_size=feature_size
            ),
            contextualizer=MeanFeatureContextualizer(
                input_size=feature_size, output_size=feature_size
            ),
        )

    @classmethod
    def construct_identity_feature_encoder(
        cls: Type["ObservationFeaturesEncoder"], feature_size: int
    ) -> "ObservationFeaturesEncoder":
        return cls(
            projector=IdentityFeatureProjector(
                input_size=feature_size, output_size=feature_size
            ),
            contextualizer=FeatureContextualizer(
                input_size=feature_size, output_size=feature_size
            ),
        )

    @classmethod
    def construct_contextualize_and_project_feature_encoder(
        cls: Type["ObservationFeaturesEncoder"],
        projector: FeatureProjector,
        contextualizer: FeatureContextualizer,
    ) -> "ObservationFeaturesEncoder":
        assert contextualizer.get_output_dim() == projector.get_input_dim()
        return cls(projector=projector, contextualizer=contextualizer, reverse=True)


ObservationFeaturesEncoder.register("project-and-contextualize")(
    ObservationFeaturesEncoder
)
ObservationFeaturesEncoder.register(
    "contextualize-and-project", "construct_contextualize_and_project_feature_encoder"
)(ObservationFeaturesEncoder)
ObservationFeaturesEncoder.register("mean", "construct_mean_feature_encoder")(
    ObservationFeaturesEncoder
)
ObservationFeaturesEncoder.register("identity", "construct_identity_feature_encoder")(
    ObservationFeaturesEncoder
)
