"""Unit tests for the model base classes."""

from collections import OrderedDict

import pytest
import torch
from torch import nn

from multitasking.models.base import ImageModel


class DummyLayer(nn.Module):
    """Dummy layer that always returns the given features."""

    def __init__(self, features: torch.Tensor):
        super().__init__()
        self.features = features

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.features


class DummyImageModel(ImageModel):
    """Image model with only dummy layers."""

    def __init__(self, features: dict[str, torch.Tensor]):
        super().__init__("dummy/dummy")
        self.base_model = nn.Sequential(OrderedDict({
            name: DummyLayer(features[name])
            for name in features
        }))

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)


def test_image_model_correctly_reshapes_cnn_features():
    video = torch.randn(2, 3, 10, 64, 64)  # B 3 T H W
    features = torch.randn(20, 32, 8, 8)  # (B T) C H' W'

    model = DummyImageModel({ "layer": features })
    extracted_features = model.extract_video_features(video, ["layer"])

    assert extracted_features["layer"].shape == (2, 32, 10, 8, 8)


def test_image_model_correctly_reshapes_transformer_features():
    video = torch.randn(2, 3, 10, 64, 64)  # B 3 T H W
    features = torch.randn(20, 8, 32)  # (B T) N C

    model = DummyImageModel({ "layer": features })
    extracted_features = model.extract_video_features(video, ["layer"])

    assert extracted_features["layer"].shape == (2, 32, 10, 8, 1)


def test_image_model_raises_error_for_unexpected_feature_shapes():
    video = torch.randn(2, 3, 10, 64, 64)  # B 3 T H W
    features = torch.randn(20, 32, 8, 8, 8)  # (B T) C H W H' W'

    model = DummyImageModel({ "layer": features })
    with pytest.raises(ValueError):
        model.extract_video_features(video, ["layer"])


@pytest.mark.parametrize(
    "postprocessed_feature_shape",
    [
        (2, 8, 32),            # Too few dimensions
        (2, 32, 8, 8),         # Too few dimensions
        (2, 32, 10, 8, 8, 8),  # Too many dimensions
        (3, 32, 10, 8, 8),     # Wrong batch dimension
        (2, 32, 9, 8, 8),      # Wrong number of frames
    ]
)
def test_image_model_raises_error_for_invalid_postprocessed_feature_shapes(
    postprocessed_feature_shape: tuple[int, int, int, int, int],
):
    video = torch.randn(2, 3, 10, 64, 64)  # B 3 T H W
    features = torch.randn(20, 32, 8, 8)  # (B T) C H W

    class DummyImageModelWithInvalidPostprocessing(DummyImageModel):
        def postprocess(
            self,
            layer: str,
            video_shape: tuple[int, int, int, int, int],
            feature: torch.Tensor,
        ) -> torch.Tensor:
            return torch.randn(*postprocessed_feature_shape)

    model = DummyImageModelWithInvalidPostprocessing({ "layer": features })
    with pytest.raises(ValueError):
        model.extract_video_features(video, ["layer"])
