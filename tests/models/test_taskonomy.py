"""Unit tests for the Taskonomy moedels."""

import pytest
import torch

from multitasking.models import taskonomy


@pytest.mark.parametrize("model_name", taskonomy.list_models())
def test_taskonomy_model_loading(model_name: str):
    model = taskonomy.build_model(model_name)
    assert model is not None
    assert model.encoder is not None


@pytest.mark.parametrize("layer", [
    "encoder",
    "encoder.layer2",
    "encoder.layer3.2",
    "encoder.layer4.0.conv3",
])
def test_taskonomy_feature_shape(layer: str):
    model = taskonomy.build_model("depth_euclidean")
    videos = torch.randn(2, 3, 10, 224, 224)
    features = model.extract_video_features(videos, [layer])

    # For each layer, we expect a single feature map with shape (B, C', T, H', W').
    # The number of batches and frames should match the input.
    assert len(features) == 1
    assert layer in features
    assert len(features[layer].shape) == 5
    assert features[layer].shape[0] == 2
    assert features[layer].shape[2] == 10
