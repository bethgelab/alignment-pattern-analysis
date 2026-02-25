"""Taskonomy (https://github.com/alexsax/visual-prior)."""

import torch
from torch import nn
from torchvision.transforms.functional import rgb_to_grayscale
from visualpriors.taskonomy_network import (
    TASKONOMY_PRETRAINED_URLS,
    TASKS_TO_CHANNELS,
    TaskonomyDecoder,
    TaskonomyEncoder,
)

from .base import ImageModel


def build_model(name: str) -> ImageModel:
    """Builds a model from the Taskonomy model bank.

    Parameters:
        name: The name of the model to build, without the provider prefix. For example,
            "depth_euclidean".

    Returns:
        The model instance.
    """
    return TaskonomyModel(name)


def list_models() -> list[str]:
    """Returns a list of all available models."""
    return [
        key.split("_encoder")[0]
        for key in TASKONOMY_PRETRAINED_URLS.keys()
        if key.endswith("_encoder")
    ]


class TaskonomyModel(ImageModel):
    """A model from the Taskonomy model bank."""

    def __init__(self, name: str):
        """Initializes the model.

        Parameters:
            name: The name of the model to build, without the provider prefix. For
                example, "depth_euclidean".
        """
        super().__init__(f"taskonomy/{name}")

        if f"{name}_encoder" not in TASKONOMY_PRETRAINED_URLS:
            raise ValueError(f"Taskonomy model for task {name} not found.")

        self.encoder = TaskonomyEncoder()

        # The colorization task only uses a single input channel.
        if name == "colorization":
            self.encoder.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=self.encoder.conv1.out_channels,
                kernel_size=self.encoder.conv1.kernel_size,
                stride=self.encoder.conv1.stride,
                padding=self.encoder.conv1.padding,
                bias=self.encoder.conv1.bias,
            )

        encoder_checkpoint = torch.utils.model_zoo.load_url( # type: ignore[attr-defined]
            TASKONOMY_PRETRAINED_URLS[f"{name}_encoder"]
        )
        self.encoder.load_state_dict(encoder_checkpoint["state_dict"])

        # For the classification tasks, the decoder checkpoint does not fit the
        # architecture defined by the TaskonomyDecoder class.
        ignore_decoder = name in ["class_object", "class_scene"]

        if ignore_decoder or f"{name}_decoder" not in TASKONOMY_PRETRAINED_URLS:
            self.decoder = None
        else:
            self.decoder = TaskonomyDecoder(TASKS_TO_CHANNELS[name])
            decoder_checkpoint = torch.utils.model_zoo.load_url( # type: ignore[attr-defined]
                TASKONOMY_PRETRAINED_URLS[f"{name}_decoder"]
            )
            self.decoder.load_state_dict(decoder_checkpoint["state_dict"])

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

        if self.name == "taskonomy/colorization":
            images = rgb_to_grayscale(images)

        images = images.to(self.device)

        # The taskonomy models expect values in the range [-1, 1].
        # https://github.com/alexsax/midlevel-reps
        images = 2 * images - 1

        output = self.encoder(images)
        if self.decoder is not None:
            output = self.decoder(output)

        return output
