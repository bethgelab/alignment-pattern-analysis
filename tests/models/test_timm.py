"""Unit tests for the timm models."""

import pytest
import torch

from multitasking.models import timm


@pytest.mark.parametrize("model_name", [
    "resnet18",
    "resnet50",
])
def test_timm_model_loading(model_name: str):
    model = timm.build_model(model_name)
    assert model is not None


@pytest.mark.parametrize("layer", [
    "layer4",
    "layer3.1",
    "layer2.0.conv2",
])
def test_timm_model_feature_shape(layer: str):
    model = timm.build_model("resnet18")
    videos = torch.randn(2, 3, 10, 224, 224)
    features = model.extract_video_features(videos, [layer])

    # For each layer, we expect a single feature map with shape (B, C', T, H', W').
    # The number of batches and frames should match the input.
    assert len(features) == 1
    assert layer in features
    assert len(features[layer].shape) == 5
    assert features[layer].shape[0] == 2
    assert features[layer].shape[2] == 10
