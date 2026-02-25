"""Unit tests for the CLIP models."""


import pytest
import torch

from multitasking.models import clip


@pytest.mark.parametrize("model_name", clip.list_models())
def test_clip_model_loading(model_name: str):
    model = clip.build_model(model_name)
    assert model is not None
    assert model.image_encoder is not None


@pytest.mark.parametrize("layer", [
    "image_encoder",
    "image_encoder.layer2",
    "image_encoder.layer3.2",
    "image_encoder.layer4.0.conv3",
])
def test_clip_resnet_feature_shape(layer: str):
    model = clip.build_model("RN50")
    videos = torch.randn(2, 3, 10, 224, 224)
    features = model.extract_video_features(videos, [layer])
    assert len(features) == 1
    assert layer in features
    assert len(features[layer].shape) == 5
    assert features[layer].shape[0] == 2
    assert features[layer].shape[2] == 10


@pytest.mark.parametrize("layer", [
    "image_encoder",
    "image_encoder.transformer.resblocks.0",
    "image_encoder.transformer.resblocks.11",
])
def test_clip_vit_feature_shape(layer: str):
    model = clip.build_model("ViT-B/32")
    videos = torch.randn(2, 3, 10, 224, 224)
    features = model.extract_video_features(videos, [layer])
    assert len(features) == 1
    assert layer in features
    assert len(features[layer].shape) == 5
    assert features[layer].shape[0] == 2
    assert features[layer].shape[2] == 10
