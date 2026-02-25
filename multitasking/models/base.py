"""Base functionality for all models."""

import logging
from abc import ABC, abstractmethod
from typing import Sequence

import einops
import torch
from torch import nn

LOGGER = logging.getLogger(__name__)




class Model(ABC, nn.Module):
    """Model base class."""

    def __init__(self, name: str):
        """Initializes the model.

        Parameters:
            name: Full name of the model, including the provider prefix.
        """
        super().__init__()

        if "/" not in name:
            raise ValueError(
                "Model name must contain a provider prefix, e.g. 'timm/resnet18'."
            )
        self.name = name

    @property
    def device(self) -> torch.device:
        """Returns the device of the model."""
        return next(self.parameters()).device

    @property
    def layers(self) -> list[str]:
        """Returns the names of all layers in the model."""
        base_model = getattr(self, "base_model", self)
        return [name for name, _ in base_model.named_modules()]

    @abstractmethod
    def extract_video_features(
        self,
        video: torch.Tensor,
        layers: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        """Extracts features from a video.

        Parameters:
            video: Video to extract features from as tensor of shape (B, C, T, H, W).
                Accepts float32 in the range [0, 1] or uint8.
            layers: Layers for which to extract features.

        Returns:
            Dictionary with the requested layer outputs.
        """
        pass


class ImageModel(Model):
    """Model that processes individual images.

    This is a base class for all image models which are applied to each video frame
    independently.
    """



    def extract_video_features(
        self,
        video: torch.Tensor,
        layers: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        """Extracts features from a video by treating each frame as a separate image.

        Parameters:
            video: Video to extract features from as tensor of shape (B, C, T, H, W).
                Accepts float32 in the range [0, 1] or uint8.
            layers: Layers for which to extract features.

        Returns:
            Dictionary with the requested layer outputs.
        """
        video_shape = video.shape
        video = einops.rearrange(video, "B C T H W -> (B T) C H W")

        base_model = getattr(self, "base_model", self)
        with LayerOutputRecorder(base_model, layers) as recorder:
            self.forward(video)
            features = recorder.outputs

        features = {
            layer: self.postprocess(layer, video_shape, feature)
            for layer, feature in features.items()
        }

        self._check_feature_shapes(video_shape, features)

        return features

    def postprocess(
        self,
        layer: str,
        video_shape: torch.Size,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """Postprocesses the features extracted from a layer.

        Per default, this method assumes 4D features for CNNs and 3D features for
        transformers and applies the following postprocessing:
        - 4D features: (B T) C H W -> B C T H W
        - 3D features: (B T) N C -> B C T N 1

        While this is a good default, models might use different conventions internally.
        In this case, the model adapter should implement custom postprocessing by
        overriding this method.

        Parameters:
            layer: Name of the layer to postprocess.
            video_shape: Shape of the input video as tuple (B, C_in, T, H_in, W_in).
            feature: Features extracted from the layer as tensor of arbitrary shape.

        Returns:
            Postprocessed features with shape (B, C_out, T, H_out, W_out).
        """
        B, _, T, _, _ = video_shape

        if feature.ndim == 3:
            return einops.rearrange(feature, "(B T) N C -> B C T N 1", B=B, T=T)

        elif feature.ndim == 4:
            return einops.rearrange(feature, "(B T) C H W -> B C T H W", B=B, T=T)

        else:
            raise ValueError(
                f"Unexpected feature shape {feature.shape} for layer {layer}. Please "
                "implement a custom postprocessing method for this model."
            )

    def _check_feature_shapes(
        self,
        video_shape: torch.Size,
        features: dict[str, torch.Tensor],
    ) -> None:
        """Checks the shapes of the features extracted from the layers."""
        B, _, T, _, _ = video_shape

        for layer, feature in features.items():
            if feature.ndim != 5:
                raise ValueError(
                    f"Unexpected feature shape {feature.shape} for layer {layer}. "
                    "Postprocessed features should be 5D, i.e. (B, C, T, H, W)."
                )
            if feature.shape[0] != B or feature.shape[2] != T:
                raise ValueError(
                    f"Batch dimension mismatch for layer {layer}. Got features with "
                    f"shape {feature.shape}, expected (B, C, T, H, W) with B={B} and "
                    f"T={T}."
                )



class LayerOutputRecorder:
    """Context manager for recording intermediate layer outputs from a model.

    Usage example:
    ```python
    with LayerOutputRecorder(model, ["block1.conv1", "block2.relu2"]) as recorder:
        model(input)
        outputs = recorder.outputs
    ```
    """

    def __init__(self, model: nn.Module, layers: Sequence[str]):
        """Initializes the recorder.

        Parameters:
            model: PyTorch model to record outputs from.
            layers: Names of the layers for which to record outputs.
        """
        self.model = model
        self.hooks = []
        self._outputs: dict[str, torch.Tensor] = {}

        def _create_hook(layer):
            def hook(module, input, output):
                self._outputs[layer] = output

            return hook

        for layer in layers:
            module = model.get_submodule(layer)
            hook = module.register_forward_hook(_create_hook(layer))
            self.hooks.append(hook)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self.hooks:
            hook.remove()

    @property
    def outputs(self) -> dict[str, torch.Tensor]:
        return self._outputs
