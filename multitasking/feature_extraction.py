"""Feature extraction stage.

This module contains all code related to extracting model features.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from benedict import benedict
from pydantic import BaseModel
from torch.nn import functional as F
from tqdm import tqdm

from multitasking.datasets import BoldMoments
from multitasking.models import build_model, get_default_layers

LOGGER = logging.getLogger(__name__)


def extract_model_representations(
    config: benedict,
    dataset: BoldMoments,
    path: Path,
    reuse: bool = False,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Extracts the representations of the model for the given config.

    Parameters:
        config: The config to use for the feature extraction.
        sample_ids: The sample IDs to extract features for.
        path: The path to the directory in which to store the features.
        reuse: If True, the features will be reused if they already exist.
    """
    if reuse and path.exists():
        LOGGER.warning("Reusing existing features from %s", path)
        return FeatureBuffer(path)  # type: ignore
    elif path.exists():
        shutil.rmtree(path)

    LOGGER.info("Building feature extractor...")
    feature_extractor = FeatureExtractor.from_config(config)

    return feature_extractor(dataset, path)  # type: ignore


class FeatureBuffer:
    """Memory-mapped buffer for model features.

    This class stores model features for a set of samples. For each layer, the features
    are provided as numpy arrays of shape (N, T, H, W, C) and dtype float32. This class
    adds some convenience methods for appending and accessing the features.
    """

    def __init__(self, path: Path, num_samples: int | None = None):
        """Initializes the feature buffer.

        Parameters:
            path: The path to the directory in which to store the features. If this
                directory does not exist, an empty buffer will be created.
            num_samples: The number of samples in the buffer. Can be omitted if loading
                an existing buffer.
        """
        self._path = path
        self._data = dict()

        if self._path.exists():
            LOGGER.info("Loading existing feature buffer from %s", self._path)

            if self._path.is_file():
                raise ValueError(f"Feature buffer path {self._path} is a file")

            if not (self._path / "shapes.json").exists():
                raise ValueError(
                    f"Feature buffer path {self._path} does not contain a shapes.json "
                    "file."
                )

            with open(self._path / "shapes.json", "r") as f:
                shapes = json.load(f)

            for layer, shape in shapes.items():
                self._data[layer] = np.memmap(
                    self._path / layer,
                    dtype=np.float32,
                    mode="r",
                    shape=tuple(shape),
                )

            num_samples_in_path = len(self._data[next(iter(self._data))])
            if num_samples is not None and num_samples != num_samples_in_path:
                raise ValueError(
                    f"Number of samples in buffer ({num_samples_in_path}) does not "
                    f"match number of samples provided ({num_samples})."
                )
            self._num_samples = num_samples_in_path
            self._next_index = num_samples_in_path

        else:
            LOGGER.info("Creating new feature buffer at %s", self._path)
            if num_samples is None:
                raise ValueError(
                    "Number of samples must be provided if creating a new buffer."
                )
            self._path.mkdir(parents=True)

            self._num_samples = num_samples
            self._next_index = 0

    @property
    def num_samples(self) -> int:
        """Returns the number of samples."""
        return self._num_samples

    @property
    def num_layers(self) -> int:
        """Returns the number of layers."""
        return len(self._data)

    @property
    def layers(self) -> list[str]:
        """Returns the available layer names."""
        return list(self._data.keys())

    def __getitem__(self, key: str) -> np.ndarray:
        """Returns the features for the given layer name.

        Returns:
            A memory-mapped numpy array of shape (N, T, H, W, C) and dtype float32.
        """
        return self._data[key]

    def append(self, data: dict[str, np.ndarray], flush: bool = True) -> None:
        """Appends the data for a single sample to the Features object.

        Parameters:
            data: A dictionary of numpy arrays, one for each layer. Each numpy array
                should have shape (T, H, W, C).
            flush: If True, the data will be flushed to disk immediately.
        """
        if any(not isinstance(feature, np.ndarray) for feature in data.values()):
            raise ValueError("All features must be numpy arrays")

        if any(layer.dtype != np.float32 for layer in data.values()):
            raise ValueError("All features must have dtype float32")

        if len(self._data) == 0:
            shapes = {
                layer: (self._num_samples, *feature.shape)
                for layer, feature in data.items()
            }
            with open(self._path / "shapes.json", "w") as f:
                json.dump(shapes, f)

            for layer, shape in shapes.items():
                self._data[layer] = np.memmap(
                    self._path / layer,
                    dtype=np.float32,
                    mode="w+",
                    shape=shape,
                )
                self._data[layer].fill(np.nan)

        if not self._data.keys() == data.keys():
            raise ValueError("All layers must be present in the Features object")

        for layer, feature in data.items():
            self._data[layer][self._next_index] = feature

        self._next_index += 1

        if flush:
            self.flush()

    def append_batch(self, data: dict[str, np.ndarray], flush: bool = True) -> None:
        """Appends the data for multiple samples to the Features object.

        Parameters:
            data: A dictionary of numpy arrays, one for each layer. Each numpy array
                should have shape (N, T, H, W, C).
            flush: If True, the data will be flushed to disk immediately.
        """
        batch_size = data[next(iter(data))].shape[0]
        for sample_index in range(batch_size):
            sample_data = {layer: data[layer][sample_index] for layer in data}
            self.append(sample_data, flush=False)

        if flush:
            self.flush()

    def flush(self) -> None:
        """Flushes the data to disk."""
        for layer in self._data:
            self._data[layer].flush()

    def validate(self) -> None:
        """Validates the Features object.

        This will run several checks to ensure that the Features object is valid and
        complete. This is useful to run after appending all samples to the Features
        object.
        """
        if len(self._data) == 0:
            raise ValueError("No features found in Features object")

        for layer, feature in self._data.items():
            if feature.ndim != 5:
                raise ValueError(
                    f"Feature for layer {layer} has {feature.ndim} dimensions, but "
                    f"expected 5 dimensions."
                )

            if np.isnan(feature).any():
                raise ValueError(f"NaNs found in feature for layer {layer}")

    def summary(self) -> str:
        """Returns a summary of the Features object."""
        summary = ""
        for layer, feature in self._data.items():
            summary += f"- {layer}: {feature.shape}\n"
        return summary


class FeatureExtractionConfig(BaseModel):
    """Configuration for the feature extraction stage."""

    model: str
    """The model to use for feature extraction."""

    layers: Sequence[str] | None = None
    """The layers to extract features from. If not provided, default layers will be
    used. If none are available, an error will be raised.
    """

    device: Literal["auto", "cpu", "cuda"] = "auto"
    """The device to run the model on."""

    batch_size: int = 8
    """The batch size to use for the dataloader."""

    num_workers: int = 8
    """The number of workers to use for the dataloader."""

    num_frames: int | None = None
    """The number of frames to postprocess the features to."""

    resolution: int | tuple[int, int] | None = None
    """The resolution to resize the feature maps to."""


class FeatureExtractor:
    """Extracts model features from a dataset."""

    def __init__(
        self,
        model: torch.nn.Module,
        layers: Sequence[str],
        device: Literal["auto", "cpu", "cuda"] = "auto",
        batch_size: int = 8,
        num_workers: int = 8,
        num_frames: int | None = None,
        resolution: int | tuple[int, int] | None = None,
    ) -> None:
        """Initializes the FeatureExtractor.

        Parameters:
            model: The model to extract features from.
            layers: The layers to extract features from.
            device: The device to run the model on. If "auto" (device), this will run
                on the first available GPU if it exists and falls back to CPU otherwise.
            batch_size: The batch size to use for the dataloader.
            num_workers: The number of workers to use for the dataloader.
            num_frames: If specified, the predicted features will be postprocessed to
                have this many frames by binning.
            resolution: If specified, the video feature maps will be resized to this
                resolution.
        """
        self.model = model
        self.layers = layers

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            LOGGER.info("Using device: %s (auto)", self.device)
        else:
            self.device = torch.device(device)
            LOGGER.info("Using device: %s", self.device)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_frames = num_frames
        if isinstance(resolution, int):
            self.resolution = (resolution, resolution)
        else:
            self.resolution = resolution  # type: ignore

        self.model.to(self.device)

    @staticmethod
    def from_config(config: benedict | FeatureExtractionConfig) -> "FeatureExtractor":
        """Builds a FeatureExtractor from a config."""
        if isinstance(config, dict):
            config = FeatureExtractionConfig(**config["feature_extraction"])

        if config.layers is None:
            default_layers = get_default_layers(config.model)
            if default_layers is None:
                raise ValueError(
                    f"No default layers found for model {config.model}. Please provide "
                    "a list of layers to extract features from."
                )
            LOGGER.info(
                "Using default layers for model %s: %s", config.model, default_layers
            )
            config.layers = default_layers
        model = build_model(config.model)

        kwargs = {k: v for k, v in config.model_dump().items() if k != "model"}

        return FeatureExtractor(model, **kwargs)

    @torch.no_grad()
    def __call__(self, dataset: BoldMoments, path: Path) -> FeatureBuffer:
        """Extracts the features of the model for the given dataset.

        Parameters:
            dataset: The dataset to extract features from.
            path: The path to the directory in which to store the features.
        """
        LOGGER.info("Extracting features...")

        self.model.eval()

        buffer = FeatureBuffer(path, len(dataset))  # type: ignore
        for index, (video, _) in tqdm(
            enumerate(dataset.load_videos(batch_size=self.batch_size)),
            total=np.ceil(len(dataset)/self.batch_size),
        ):
            if index == 0:
                LOGGER.info("Video shape: %s", video.shape)
            features = self.model.extract_video_features(video, self.layers)  # type: ignore  # noqa: E501

            if index == 0:
                LOGGER.info("Feature shapes before postprocessing (NCTHW):")
                for layer, feature in features.items():
                    LOGGER.info("- %s: %s", layer, feature.shape)

            features = self._postprocess(features)

            if index == 0:
                LOGGER.info("Feature shapes after postprocessing (NTHWC):")
                for layer, feature in features.items():
                    LOGGER.info("- %s: %s", layer, feature.shape)

            buffer.append_batch(features)

        buffer.validate()

        LOGGER.info("Extracting features done.")
        LOGGER.info(buffer.summary())

        return buffer

    def _postprocess(self, features: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        """Postprocess the model features."""
        for layer, feature in features.items():
            if self.resolution is not None:
                T, H, W = feature.shape[-3:]
                if H > self.resolution[0] and W > self.resolution[1]:
                    feature = F.adaptive_avg_pool3d(feature, (T, *self.resolution))
                elif H < self.resolution[0] or W < self.resolution[1]:
                    feature = F.interpolate(
                        feature,
                        (T, *self.resolution),
                        mode="trilinear" if feature.ndim == 5 else "bilinear",
                        align_corners=False,
                    )
                else:
                    feature = F.interpolate(
                        feature,
                        (T, *self.resolution),
                        mode="bilinear",
                        align_corners=False,
                    )
                # Weird that the above seems to work, aren't the dimensions wrong?
                # elif H < self.resolution[0] or W < self.resolution[1]:
                #     feature = F.interpolate(
                #         feature,
                #         (T, *self.resolution),
                #         mode="trilinear" if feature.ndim == 5 else "bilinear",
                #         align_corners=False,
                #     )

            features[layer] = feature.cpu().numpy()  # type: ignore

            # NCHWT -> NTHWC
            features[layer] = np.moveaxis(features[layer], 1, -1)  # type: ignore  # noqa: E501

            if self.num_frames is not None:
                _, features[layer] = _downsample_to_length(
                    features[layer], self.num_frames
                )

        return features  # type: ignore


def _downsample_to_length(data, target_len):
    """Downsample an array to a new length by averaging over bins.

    Parameters:
    - data: Input array with arbitrary frame axis.
    - target_len (int): Target length after downsampling.

    Returns:
    - bins (ndarray): Bin edges used for downsampling.
    - result (ndarray): Downsampled array with the same number of dimensions as input.
    """
    original_len = data.shape[1]
    assert original_len >= target_len, (
        "Target length must be less than or equal to input length"
    )

    # Compute bin edges
    bins = np.linspace(0, original_len, target_len + 1, dtype=int)

    result = np.stack(
        [data[:, bins[i] : bins[i + 1], ...].mean(axis=1) for i in range(target_len)],
        axis=1,
    )

    return bins, result
