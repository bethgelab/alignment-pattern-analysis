"""Unit tests for the MMAction2 models."""

import pytest
import torch

pytest.importorskip("mmaction")

from multitasking.models import mmaction2


@pytest.mark.parametrize("model_name", mmaction2.list_models())
def test_mmflow_model_loading(model_name: str):
    model = mmaction2.build_model(model_name)
    assert model is not None


@pytest.mark.parametrize("layer", [
    "backbone.blocks.1",
    "backbone.blocks.14",
])
def test_mmaction2_mvit_feature_shape(layer: str):
    model = mmaction2.build_model("mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb")
    videos = torch.randn(1, 3, 10, 224, 224)
    features = model.extract_video_features(videos, [layer])

    # For each layer, we expect a single feature map with shape (B, C', T=1, H', W').
    # The number of batches should match the input, but we expect the model to collapse
    # the temporal dimension.
    assert len(features) == 1
    assert layer in features
    assert len(features[layer].shape) == 5
    assert features[layer].shape[0] == 1
    assert features[layer].shape[2] == 1


@pytest.mark.parametrize("layer", [
    "backbone.layers.1",
    "backbone.layers.2.blocks.0",
    "backbone.layers.3.blocks.1.norm2",
])
def test_mmaction2_swin_tiny_feature_shape(layer: str):
    model = mmaction2.build_model(
        "swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb"
    )
    videos = torch.randn(1, 3, 10, 224, 224)
    features = model.extract_video_features(videos, [layer])

    # For each layer, we expect a single feature map with shape (B, C', T', H', W').
    # The Swin Transformer however keeps the spatial structure of the patches, so we
    # should recover meaningful T, H, W dimensions. The size of the time timension T
    # depends on the clip size used during preprocessing and the length of the input
    # video. So we just check that time is not collapsed, but not for the exact length.
    assert len(features) == 1
    assert layer in features
    assert len(features[layer].shape) == 5
    assert features[layer].shape[0] == 1
    assert features[layer].shape[2] > 1  # time dimension not collapsed


@pytest.mark.parametrize("layer", [
    "backbone.transformer_layers.layers.1",
    "backbone.transformer_layers.layers.11",
])
def test_mmaction2_timesformer_feature_shape(layer: str):
    model = mmaction2.build_model(
        "timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb"
    )
    videos = torch.randn(1, 3, 10, 224, 224)
    features = model.extract_video_features(videos, [layer])

    # For each layer, we expect a single feature map with shape (B, C', T=1, H', W').
    # The number of batches should match the input, but we expect the model to collapse
    # the temporal dimension.
    assert len(features) == 1
    assert layer in features
    assert len(features[layer].shape) == 5
    assert features[layer].shape[0] == 1
    assert features[layer].shape[2] == 1
