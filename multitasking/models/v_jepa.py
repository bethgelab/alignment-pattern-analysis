"""V_JEPA https://github.com/facebookresearch/vjepa2."""

import logging
from typing import Sequence

import torch

from .base import LayerOutputRecorder, Model

LOGGER = logging.getLogger(__name__)



IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)



def build_model(name: str) -> Model:
    """Builds a V-JEPA2 model.

    Parameters:
        name: The name of the model to build, without the provider prefix. For example,
            "vjepa2_vit_large".

    Returns:
        The model instance.
    """
    return V_JEPA(name)


def list_models() -> list[str]:
    """Returns a list of all available models."""
    model_list = ['vjepa2_vit_large', 'vjepa2_vit_huge', 'vjepa2_vit_giant',
                  'vjepa2_vit_giant_384']
    return model_list


class V_JEPA(Model):
    """A V-JEPA2 model."""

    def __init__(self, name: str):
        """Initializes the model.

        Parameters:
            name: The name of the model to build, without the provider prefix. For
                example, "vjepa2_vit_large".
        """
        super().__init__(f"v_jepa/{name}")
        encoder, projector = torch.hub.load('facebookresearch/vjepa2', name,
                                             pretrained=True)
        self.encoder = encoder
        self.projector = projector
        # register encoder and projector as submodules without dot in names
        self.base_model = encoder

        LOGGER.info(f"Loaded V_JEPA model: {name}")
        LOGGER.info("model structure: %s", self.base_model)

    @property
    def layers(self) -> list[str]:
        """Returns the names of layers available for feature extraction."""
        return [name for name, _ in self.base_model.named_modules()]

    def extract_video_features(
        self,
        video: torch.Tensor,
        layers: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        """Extracts features from a video."""
        """Input is (1,C,T,H,W)"""
        LOGGER.info("video shape: %s", video.shape)

        video = video.movedim(2, 1).flatten(0, 1)  # (B * T, C, H, W)

        with LayerOutputRecorder(self.base_model, layers) as recorder:
            self.forward(video)
            features = recorder.outputs


        for layer, feature in features.items():
            try:
                feature = feature.unsqueeze(-1).unsqueeze(-1).contiguous()
                # (B, F, C, T, 1) -> (B, C, T, F, 1)
                feature = feature.permute(0, 2, 3, 1, 4).contiguous()
            except ValueError as e:
                LOGGER.error(f"Layer {layer}: {e}")
                raise
            features[layer] = feature

        return features

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

        # we want # T x C x H x W because B is one
        images = images.unsqueeze(0)
        #change dims 1,2
        images = images.movedim(1, 2)

        return self.base_model(images)
