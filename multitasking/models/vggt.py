"""VGG-Transformer (https://github.com/facebookresearch/vggt)."""

import math
import tempfile
import warnings
from pathlib import Path
from typing import Sequence

import PIL.Image
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

from .base import LayerOutputRecorder, Model


def build_model(name: str) -> Model:
    """Builds a model from the VGG-Transformer library.

    Parameters:
        name: The name of the model to build, without the provider prefix. For example,
            "VGGT-1B".
    """
    return VGGTransformer(name)


def list_models() -> list[str]:
    """Returns a list of all available models."""
    return ["VGGT-1B"]


class VGGTransformer(Model):
    """VGG-Transformer model (VGGT).

    The VGGT consists of an encoder (named "aggregator") that extracts features from
    a set of input images. Different prediction heads are trained to predict camera
    parameters, depth, point maps and tracking features. This adapter allows extracting
    features from the encoder.
    """

    def __init__(self, name: str = "VGGT-1B"):
        """Initializes the VGG-Transformer model.

        Parameters:
            name: The name of the model to load, without the provider prefix. For
                example, "VGGT-1B".
        """
        super().__init__(f"vggt/{name}")
        self.base_model = VGGT.from_pretrained(f"facebook/{name}")

        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
        self.dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        # We will only use the aggregator (encoder), so we can remove all heads and save
        # some resources.
        self.base_model.camera_head = None
        self.base_model.depth_head = None
        self.base_model.point_head = None
        self.base_model.track_head = None

    @property
    def layers(self) -> list[str]:
        """Returns the names of the layers available for feature extraction."""
        aggregator = self.base_model.aggregator

        if self.base_model.aggregator.aa_block_size != 1:
            warnings.warn(
                "The AA block size of VGGT is not 1, which means that the layer names "
                "are not listed in the same order as they are executed.",
                stacklevel=2,
            )

        layers = []
        for layer_index in range(aggregator.depth):
            for block_type in aggregator.aa_order:
                layers.append(f"aggregator.{block_type}_blocks.{layer_index}")

        return [*layers, "aggregator"]

    def extract_video_features(
        self,
        video: torch.Tensor,
        layers: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        """Extracts features from a video by treating each frame as a separate image.

        Parameters:
            video: Video to extract features from as tensor of shape (B, C, T, H, W).
                Accepts float32 in the range [0, 1] or uint8. This only supports a batch
                size of B=1.
            layers: Layers for which to extract features.

        Returns:
            Dictionary with the requested layer outputs.
        """
        _, _, T, H, W = video.shape
        if H != W:
            raise NotImplementedError(
                "Feature extraction for the VGG-Transformer only supports square input "
                "resolutions (H=W)."
            )

        with LayerOutputRecorder(self.base_model, layers) as recorder:
            self.forward(video)
            outputs = recorder.outputs

        features = dict()
        patch_start_idx = self.base_model.aggregator.patch_start_idx

        for layer_name, layer_outputs in outputs.items():
            if layer_name == "aggregator":
                patch_tokens = layer_outputs[0][-1]
            elif "patch_embed" in layer_name:
                patch_tokens = layer_outputs.unsqueeze(0)  # T P C -> B=1 T P C
            elif "frame_blocks" in layer_name:
                patch_tokens = layer_outputs.unsqueeze(0)  # BT P C -> B=1 T P C
            elif "global_blocks" in layer_name:
                B, _, C = layer_outputs.shape
                patch_tokens = layer_outputs.view(B, T, -1, C)  # B T P C -> B T P C

            # Remove all special tokens, such as camera and register tokens.
            # IDEA It might be interesting to keep the special tokens, as they might
            #      contain useful features.
            patch_tokens = patch_tokens[:, :, patch_start_idx:]

            # Reshape to (B, C, T, H, W), assuming square input images.
            B, T, N, C = patch_tokens.shape
            H = W = int(math.sqrt(N))
            if H * W != N:
                raise ValueError(f"Invalid number of patch tokens: {N}")
            patch_tokens = patch_tokens.movedim(-1, 1).reshape(B, C, T, H, W)

            features[layer_name] = patch_tokens

        return features

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass of the VGG-Transformer model.

        Parameters:
            video: Video as tensor of shape (B, C, T, H, W), with values in the range
                [0, 255] for uint8 images and [0, 1] for float32 images. This only
                supports a batch size of B=1.

        Returns:
            The outputs of the VGG-Transformer model as a dictionary.
        """
        if video.shape[0] > 1:
            raise NotImplementedError(
                "For the VGG-Transformer, feature extraction only supports a batch "
                "size of 1."
            )

        if video.dtype == torch.float32:
            video = (video * 255.0).to(torch.uint8)
        else:
            video = video.to(torch.uint8)

        # The provided preprocessing function expects a list of image paths. We
        # save the video frames to a temporary directory and use the paths to
        # load the images. Not the most efficient solution, but we don't lose much
        # time here.
        with tempfile.TemporaryDirectory() as temp_dir:
            image_paths = []
            for frame_index in range(video.shape[2]):
                image_path = Path(temp_dir) / f"{frame_index}.png"
                image = PIL.Image.fromarray(
                    video[0, :, frame_index].movedim(0, -1).cpu().numpy()
                )
                image.save(image_path)
                image_paths.append(str(image_path))

            images = load_and_preprocess_images(image_paths).to(self.device)

            with torch.no_grad():
                with torch.amp.autocast(str(self.device), dtype=self.dtype):
                    return self.base_model(images)
