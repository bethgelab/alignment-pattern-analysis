"""VISSL models (https://github.com/facebookresearch/vissl)."""

import warnings

import torch.utils.model_zoo
from torch import nn
from torchvision.models.resnet import resnet50, resnet101

from .base import ImageModel

# These are only the SimCLR models with the regular ResNet-50 and ResNet-101
# architectures. More model checkpoints can be found here:
# https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md
CHECKPOINT_URLS = {
    "simclr_resnet50_100epochs": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_100ep_simclr_8node_resnet_16_07_20.8edb093e/model_final_checkpoint_phase99.torch",
    "simclr_resnet50_200epochs": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_200ep_simclr_8node_resnet_16_07_20.a816c0ef/model_final_checkpoint_phase199.torch",
    "simclr_resnet50_400epochs": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_400ep_simclr_8node_resnet_16_07_20.36b338ef/model_final_checkpoint_phase399.torch",
    "simclr_resnet50_800epochs": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1/model_final_checkpoint_phase799.torch",
    "simclr_resnet50_1000epochs": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_1000ep_simclr_8node_resnet_16_07_20.afe428c7/model_final_checkpoint_phase999.torch",
    "simclr_resnet101_100epochs": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn101_100ep_simclr_8node_resnet_16_07_20.1ff6cb4b/model_final_checkpoint_phase99.torch",
    "simclr_resnet101_1000epochs": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn101_1000ep_simclr_8node_resnet_16_07_20.35063cea/model_final_checkpoint_phase999.torch",
}


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def build_model(name: str) -> ImageModel:
    """Builds a model from the VISSL library.

    Parameters:
        name: The name of the model to build, without the provider prefix. For example,
            "vissl/simclr_resnet50_100epochs".

    """
    return ViSSLModel(name)


def list_models() -> list[str]:
    """Returns a list of all available models."""
    warnings.warn(
        "Not all VISSL models have been added to the list of available models.",
        UserWarning,
        stacklevel=2,
    )
    return list(CHECKPOINT_URLS.keys())


class ViSSLModel(ImageModel):
    """VISSL model."""

    def __init__(self, name: str):
        """Initializes the model.

        Parameters:
            name: The name of the model to build, without the provider prefix. For
                example, "simclr_resnet50_1000epochs".
        """
        super().__init__(f"vissl/{name}")

        checkpoint_url = CHECKPOINT_URLS[name]
        checkpoint = torch.utils.model_zoo.load_url(checkpoint_url)

        if "model_state_dict" in checkpoint:
            weights = checkpoint["model_state_dict"]
        elif "classy_state_dict" in checkpoint:
            weights = checkpoint["classy_state_dict"]["base_model"]["model"]["trunk"]
        else:
            raise ValueError(f"Unknown checkpoint format: {checkpoint.keys()}")

        weights = {k.replace("_feature_blocks.", ""): v for k, v in weights.items()}

        if "resnet50" in name:
            self.base_model = resnet50(pretrained=False)
        elif "resnet101" in name:
            self.base_model = resnet101(pretrained=False)

        # Remove the final classification layer, as no weights are provided for it.
        self.base_model.fc = nn.Identity()

        self.base_model.load_state_dict(weights)

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

        mean = _IMAGENET_MEAN.to(self.device)
        std = _IMAGENET_STD.to(self.device)
        images = (images - mean[None, :, None, None]) / std[None, :, None, None]

        return self.base_model(images)
