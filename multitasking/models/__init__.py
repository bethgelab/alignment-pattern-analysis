"""Models.

This package provides a unified interface for a range of models that can be used for
video feature extraction.
"""

import importlib
import warnings

from multitasking.models.base import Model

# Overall, several hundred models are availble from the different providers. The
# following list contains some of the most popular models that we include in our
# analysis.
FAVORITE_MODELS = [
    # All models from the Taskonomy model bank.
    "taskonomy/autoencoding",
    "taskonomy/class_object",
    "taskonomy/class_scene",
    "taskonomy/colorization",
    "taskonomy/curvature",
    "taskonomy/denoising",
    "taskonomy/depth_euclidean",
    "taskonomy/depth_zbuffer",
    "taskonomy/edge_occlusion",
    "taskonomy/edge_texture",
    "taskonomy/egomotion",
    "taskonomy/fixated_pose",
    "taskonomy/inpainting",
    "taskonomy/jigsaw",
    "taskonomy/keypoints2d",
    "taskonomy/keypoints3d",
    "taskonomy/nonfixated_pose",
    "taskonomy/normal",
    "taskonomy/point_matching",
    "taskonomy/reshading",
    "taskonomy/room_layout",
    "taskonomy/segment_semantic",
    "taskonomy/segment_unsup25d",
    "taskonomy/segment_unsup2d",
    "taskonomy/vanishing_point",
    # Two popular ResNets. The A1 training scheme is designed to maximize the
    # performance of the ResNet-50 model on ImageNet (https://arxiv.org/abs/2110.00476).
    "timm/resnet18.a1_in1k",
    "timm/resnet50.a1_in1k",
    # ConvNext trained on the same data (ImageNet-1K) as the ResNet models above.
    "timm/convnext_base.fb_in1k",
    "timm/convnext_large.fb_in1k",
    "timm/convnext_small.fb_in1k",
    "timm/convnext_tiny.fb_in1k",
    # Resnet-50 and Resnet-101 trained using SimCLR.
    "vissl/simclr_resnet50_1000epochs",
    "vissl/simclr_resnet101_1000epochs",
    # CLIP ResNet-50 and ViT
    "clip/RN50",
    "clip/ViT-B/32",
    "clip/ViT-B/16",
    "clip/ViT-L/14",
    "clip/ViT-L/14@336px",
    # Opt-CWM
    "opt_cwm/opt_cwm",
    # VGG-Transformer
    "vggt/VGGT-1B",
    # MMAction2 models
    "mmaction2/mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb",
    "mmaction2/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb",
    "mmaction2/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb",
    "mmaction2/timesformer_jointST_8xb8-8x32x1-15e_kinetics400-rgb",
    #VJEPA2 models
    "v_jepa/vjepa2_vit_large",
    "v_jepa/vjepa2_vit_huge",
    "v_jepa/vjepa2_vit_giant",
    "v_jepa/vjepa2_vit_giant_384"

]


def build_model(name: str) -> Model:
    """Builds the model with the given name.

    Parameters:
        name: The full name of the model "provider/model_name". The provider is the
            name of the package that contains the model implementation, the model_name
            referes to the model name within that package (e.g.,
            "timm/resnet50.a1_in1k").

    Returns:
        The model instance.
    """
    provider_name, model_name = name.split("/", maxsplit=1)
    provider = _get_provider(provider_name)
    return provider.build_model(model_name)  # type: ignore


def list_providers() -> list[str]:
    """Returns a list of all available model providers."""
    return [
        "clip",
        "mmaction2",
        "opt_cwm",
        "taskonomy",
        "timm",
        "vggt",
        "vissl",
        "v_jepa"
    ]


def _get_provider(name: str) -> Model:
    """Returns the provider with the given name."""
    if name in ["__init__", "base", "demo"]:
        raise ValueError(f"Invalid model provider: {name}")
    try:
        return importlib.import_module(f"multitasking.models.{name}")  # type: ignore
    except ImportError as error:
        raise ImportError(f"Unable to import model provider: {name}") from error


def list_models(provider: str | None = None) -> list[str]:
    """Returns a list of all available models.

    Parameters:
        provider: The provider to list models for. If None, models from all providers
            are returned.
    """
    if provider is None:
        return [
            model for provider in list_providers() for model in list_models(provider)
        ]
    else:
        models: list[str] = []
        for provider_name in list_providers():
            try:
                provider = _get_provider(provider_name)  # type: ignore
            except (ImportError, AssertionError):
                warnings.warn(
                    f"Unable to import model provider: {provider_name}",
                    stacklevel=2,
                )
                continue
            models.extend(
                f"{provider_name}/{model}" for model in provider.list_models()  # type: ignore  # noqa: E501
            )
        return models


def get_default_layers(name: str) -> list[str] | None:
    """Returns the default layers for the given model.

    Parameters:
        name: The full name of the model (e.g., "timm/resnet50.a1_in1k").

    Returns:
        The default layers for the given model. If no default layers are available for
        the model, None is returned.
    """
    if name.startswith("clip/RN"):
        return ["image_encoder.layer4"]

    if name.startswith("clip/ViT"):
        return ["image_encoder"]

    if name.startswith("opt_cwm/"):
        return ["cwm_model.encoder"]

    if name.startswith("taskonomy/"):
        return ["encoder"]

    if name.startswith("timm/resnet") and name in FAVORITE_MODELS:
        # Last feature map before the global average pooling
        return ["layer4"]

    if name.startswith("timm/convnext") and name in FAVORITE_MODELS:
        # Last feature map before the global average pooling
        return ["stages"]

    if name.startswith("vggt/"):
        return ["aggregator"]

    if name.startswith("vissl/simclr_resnet"):
        return ["layer4"]

    if name.startswith("v_jepa/"):
        return ["blocks.0.mlp.fc2"]

    return None
