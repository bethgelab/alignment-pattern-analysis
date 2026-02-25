"""Timm (https://github.com/rwightman/pytorch-image-models)."""

import timm
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .base import ImageModel


def build_model(name: str) -> ImageModel:
    """Builds a model from the Timm library.

    Parameters:
        name: The name of the model to build, without the provider prefix. For example,
            "resnet50.a1_in1k".

    Returns:
        The model instance.
    """
    return TimmModel(name)


def list_models() -> list[str]:
    """Returns a list of all available models."""
    return timm.list_models(pretrained=True)


class TimmModel(ImageModel):
    """A model from the Timm library."""

    def __init__(self, name: str):
        """Initializes the model.

        Parameters:
            name: The name of the model to build, without the provider prefix. For
                example, "resnet50.a1_in1k".
        """
        super().__init__(f"timm/{name}")
        self.base_model = timm.create_model(name, pretrained=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Applies the model to a batch of images.

        Args:
            images: Images as tensor of shape (B, C, H, W), with values in the range
                [0, 255] for uint8 images and [0, 1] for float32 images.

        Returns:
            The output of the model.
        """
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif images.dtype != torch.float32:
            raise ValueError(
                f"Input must be of type uint8 or float32, got {images.dtype}."
            )

        images = images.to(self.device)

        mean = torch.tensor(IMAGENET_DEFAULT_MEAN).to(self.device)
        std = torch.tensor(IMAGENET_DEFAULT_STD).to(self.device)
        images = (images - mean[None, :, None, None]) / std[None, :, None, None]

        return self.base_model(images)
