"""Unit tests for the VGG-Transformer model."""

import pytest
import torch

from multitasking.models import vggt


@pytest.mark.parametrize("model_name", vggt.list_models())
def test_vggt_model_loading(model_name: str):
    model = vggt.build_model(model_name)
    assert model is not None
    assert model.base_model is not None


@pytest.mark.parametrize("layer", [
    "aggregator",
    "aggregator.frame_blocks.4",
    "aggregator.global_blocks.18",
])
def test_vggt_feature_shape(layer: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = vggt.build_model("VGGT-1B").to(device)
    videos = torch.randn(1, 3, 10, 224, 224)
    features = model.extract_video_features(videos, [layer])
    assert len(features) == 1
    assert layer in features
    assert len(features[layer].shape) == 5
    assert features[layer].shape[0] == 1
    assert features[layer].shape[2] == 10
