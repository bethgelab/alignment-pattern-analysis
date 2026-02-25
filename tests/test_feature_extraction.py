"""Tests for the feature extraction module."""

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from multitasking.feature_extraction import FeatureBuffer


@pytest.fixture()
def tempdir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as tempdir:
        yield Path(tempdir)


def test_empty_feature_buffer_has_given_number_of_samples(tempdir: Path):
    features = FeatureBuffer(tempdir / "test", num_samples=10)
    assert features.num_samples == 10


def test_empty_feature_buffer_has_no_layers(tempdir: Path):
    features = FeatureBuffer(tempdir / "test", num_samples=10)
    assert features.num_layers == 0


def test_feature_buffer_append_updates_num_layers(tempdir: Path):
    features = FeatureBuffer(tempdir / "test", num_samples=10)
    features.append({"layer1": np.zeros((8, 10, 4, 4), dtype=np.float32)})
    assert features.num_layers == 1


def test_feature_buffer_append_updates_layer_names(tempdir: Path):
    features = FeatureBuffer(tempdir / "test", num_samples=10)
    features.append({"layer1": np.zeros((8, 10, 4, 4), dtype=np.float32)})
    assert set(features.layers) == {"layer1"}


def test_feature_buffer_append_aggregates_features(tempdir: Path):
    features = FeatureBuffer(tempdir / "test", num_samples=2)
    features.append({"layer1": np.zeros((8, 10, 4, 4), dtype=np.float32)})
    features.append({"layer1": np.ones((8, 10, 4, 4), dtype=np.float32)})
    assert features["layer1"].shape == (2, 8, 10, 4, 4)
    assert np.all(features["layer1"][0] == 0)
    assert np.all(features["layer1"][1] == 1)


def test_feature_buffer_append_requires_all_layers(tempdir: Path):
    features = FeatureBuffer(tempdir / "test", num_samples=2)
    data = {
        "layer1": np.zeros((8, 10, 4, 4), dtype=np.float32),
        "layer2": np.zeros((8, 10, 4, 4), dtype=np.float32),
    }
    features.append(data)
    with pytest.raises(ValueError):
        features.append({"layer1": np.zeros((8, 10, 4, 4), dtype=np.float32)})


def test_feature_buffer_raises_if_full(tempdir: Path):
    features = FeatureBuffer(tempdir / "test", num_samples=1)
    data = {
        "layer1": np.zeros((8, 10, 4, 4), dtype=np.float32),
        "layer2": np.zeros((8, 10, 4, 4), dtype=np.float32),
    }
    features.append(data)
    with pytest.raises(IndexError):
        features.append(data)


def test_empty_feature_buffer_is_not_valid(tempdir: Path):
    features = FeatureBuffer(tempdir / "test", num_samples=1)
    with pytest.raises(ValueError):
        features.validate()


def test_feature_buffer_with_nans_is_not_valid(tempdir: Path):
    features = FeatureBuffer(tempdir / "test", num_samples=1)

    data = {"layer1": np.full((1, 8, 10, 4, 4), np.nan, dtype=np.float32)}
    features.append(data)

    with pytest.raises(ValueError):
        features.validate()


def test_feature_buffer_with_to_few_samples_is_not_valid(tempdir: Path):
    features = FeatureBuffer(tempdir / "test", num_samples=2)
    features.append({"layer1": np.zeros((8, 10, 4, 4), dtype=np.float32)})
    with pytest.raises(ValueError):
        features.validate()


def test_feature_buffer_with_all_samples_is_valid(tempdir: Path):
    features = FeatureBuffer(tempdir / "test", num_samples=2)
    features.append({"layer1": np.zeros((8, 10, 4, 4), dtype=np.float32)})
    features.append({"layer1": np.ones((8, 10, 4, 4), dtype=np.float32)})
    features.validate()
