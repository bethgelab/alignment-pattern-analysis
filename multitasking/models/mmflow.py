"""MMFlow (https://github.com/open-mmlab/mmflow)."""

import os
import tempfile
from typing import Sequence
from urllib.request import urlretrieve

import numpy as np
import torch
from mmflow.apis import inference_model, init_model

from .base import LayerOutputRecorder, Model


def build_model(name: str) -> Model:
    """Builds a model from the MMFlow library.

    Parameters:
        name: The name of the model to build, without the provider prefix. For example,
            "raft_8x2_100k_mixed_368x768".

    Returns:
        The model instance.
    """
    return MMFlowModel(name)


def list_models() -> list[str]:
    """Returns a list of all available models."""
    return [
        "flownet2cs_8x1_slong_flyingchairs_384x448",
        "flownet2cs_8x1_sfine_flyingthings3d_subset_384x768",
        "flownet2css_8x1_slong_flyingchairs_384x448",
        "flownet2css_8x1_sfine_flyingthings3d_subset_384x768",
        "flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448",
        "flownet2_8x1_slong_flyingchairs_384x448_20220625_212801-88d61800",
        "flownet2_8x1_sfine_flyingthings3d_subset_384x768",
        "flownet2sd_8x1_slong_chairssdhom_384x448",
        "gma_8x2_120k_flyingchairs_368x496",
        "gma_8x2_120k_flyingthings3d_400x720",
        "gma_8x2_120k_flyingthings3d_sintel_368x768",
        "gma_8x2_120k_mixed_368x768",
        "gma_8x2_50k_kitti2015_288x960",
        "gma_p-only_8x2_120k_flyingchairs_368x496",
        "gma_p-only_8x2_120k_flyingthings3d_400x720",
        "gma_p-only_8x2_120k_mixed_368x768",
        "gma_p-only_8x2_50k_kitti2015_288x960",
        "gma_plus-p_8x2_120k_flyingchairs_368x496",
        "gma_plus-p_8x2_120k_flyingthings3d_400x720",
        "gma_plus-p_8x2_120k_mixed_368x768",
        "gma_plus-p_8x2_50k_kitti2015_288x960",
        "pwcnet_8x1_slong_flyingchairs_384x448",
        "pwcnet_8x1_sfine_flyingthings3d_subset_384x768",
        "pwcnet_ft_4x1_300k_sintel_384x768",
        "pwcnet_ft_4x1_300k_sintel_final_384x768",
        "pwcnet_ft_4x1_300k_kitti_320x896",
        "pwcnet_plus_8x1_750k_sintel_kitti2015_hd1k_320x768",
        "raft_8x2_100k_flyingchairs_368x496",
        "raft_8x2_100k_flyingthings3d_400x720",
        "raft_8x2_100k_flyingthings3d_sintel_368x768",
        "raft_8x2_100k_mixed_368x768",
        "raft_8x2_50k_kitti2015_288x960",
    ]


class MMFlowModel(Model):
    """A model from the MMFlow library."""

    def __init__(self, name: str):
        """Initializes the model.

        Parameters:
            name: The name of the model to build, without the provider prefix. For
                example, "raft_8x2_100k_mixed_368x768".
        """
        super().__init__(f"mmflow/{name}")

        with tempfile.TemporaryDirectory() as temp_dir:
            model_family = self._get_model_family(name)
            base_url = f"https://download.openmmlab.com/mmflow/{model_family}"

            config_url = f"{base_url}/{name}.py"
            config_path = os.path.join(temp_dir, "config.py")
            urlretrieve(config_url, config_path)

            checkpoint_url = f"{base_url}/{name}.pth"
            checkpoint_path = os.path.join(temp_dir, "checkpoint.pth")
            urlretrieve(checkpoint_url, checkpoint_path)

            self.base_model = init_model(config_path, checkpoint_path)

    def forward(self, image_pairs: torch.Tensor) -> list[np.ndarray] | np.ndarray:
        """Applies the model to a batch of image pairs.

        Parameters:
            image_pairs: Image pairs as tensor of shape (B, C, 2, H, W), with values in
                the range [0, 1] for float32 images and [0, 255] for uint8 images.

        Returns:
            The output of the model (optical flow).
        """
        if image_pairs.dtype == torch.float32:
            image_pairs = (image_pairs * 255.0).to(torch.uint8)
        elif image_pairs.dtype != torch.uint8:
            raise ValueError(
                f"Input must be of type uint8 or float32, got {image_pairs.dtype}."
            )

        frame0 = list(image_pairs[:, :, 0].movedim(1, -1).cpu().numpy().copy())
        frame1 = list(image_pairs[:, :, 1].movedim(1, -1).cpu().numpy().copy())

        # MMFlow takes care of moving the input tensors to the correct device.
        return inference_model(self.base_model, frame0, frame1, [None] * len(frame0))

    def extract_video_features(
        self,
        video: torch.Tensor,
        layers: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        """Extracts features from a video.

        Parameters:
            video: Video as tensor of shape (B, C, T, H, W), with values in the range
                [0, 1] for float32 images and [0, 255] for uint8 images.
            layers: Layers for which to extract features.
        """
        # For the optical flow models we need pairs of frames.
        video = torch.stack(
            [video[:, :, :-1], video[:, :, 1:]], dim=3
        )  # (B, C, T-1, 2, H, W)

        # Flatten the batch and time dimensions.
        batch_size, _, num_frames = video.shape[:3]
        video = video.movedim(2, 1).flatten(0, 1)  # (B * (T-1), C, 2, H, W)

        with LayerOutputRecorder(self.base_model, layers) as recorder:
            self.forward(video)
            features = recorder.outputs

        # Reshape the features back to the original shape.
        for layer, feature in features.items():
            feature = feature.view(batch_size, num_frames, *feature.shape[1:])
            feature = feature.movedim(1, 2)  # (B, C, T-1, H, W)
            features[layer] = feature

        return features

    @staticmethod
    def _get_model_family(model_name: str) -> str:
        if model_name.startswith("flownet2"):
            return "flownet2"
        elif model_name.startswith("flownet"):
            return "flownet"
        elif model_name.startswith("irr"):
            return "irr"
        elif model_name.startswith("maskflownet"):
            return "maskflownet"
        else:
            return model_name.split("_")[0]
