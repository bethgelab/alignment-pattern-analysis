"""Feature reduction stage."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.random_projection import (
    SparseRandomProjection,
    johnson_lindenstrauss_min_dim,
)

LOGGER = logging.getLogger(__name__)


def reduce_features(
    config: dict,
    features: npt.NDArray[np.float32],
    projector: FeatureReduction | None = None,
) -> tuple[dict[str, np.ndarray], FeatureReduction]:
    """Feature reduction.

    Parameters:
        config: The config.
        features: The features extracted from the model.
        projector: None, or an existing feature reduction object to re-use.
                   T.g. for calling this function twice for connected datasets.

    Returns:
        A dictionary of reduced features. For each layer, the reduced features are
        stored in a numpy array of shape (num_samples, num_features).
    """
    if projector is None:
        reduction = FeatureReduction.from_config(config["feature_reduction"])
    else:
        reduction = projector
    return reduction(features), reduction  # type: ignore


class FeatureReduction(ABC):
    """Feature reduction base class."""

    @abstractmethod
    def __call__(self, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Reduce the features.

        Parameters:
            features: The features to reduce with shape (num_samples, num_features).

        Returns:
            The reduced features with shape (num_samples, num_components).
        """
        pass

    @staticmethod
    def from_config(config: dict) -> "FeatureReduction":
        """Create a feature reduction object from a config.

        Parameters:
            config: The config.
        """
        kwargs = {k: v for k, v in config.items() if k != "method"}
        if config["method"] == "incremental_pca":
            # only delete when its there
            if "seed" in kwargs:
                del kwargs["seed"]
            return IncrementalPCAFeatureReduction(**kwargs)
        elif config["method"] == "pca":
            return PCAFeatureReduction(**kwargs)
        elif config["method"] == "srp":
            if config.get("n_components") is not None:
                del kwargs["n_components"]
            return SRPFeatureReduction(**kwargs)
        elif config["method"] == "none":
            return NoFeatureReduction()
        else:
            raise ValueError(f"Unknown feature reduction method: {config['method']}")


class IncrementalPCAFeatureReduction(FeatureReduction):
    """Incremental PCA feature reduction."""

    def __init__(self, n_components: int):
        """Initialize the incremental PCA feature reduction.

        Parameters:
            n_components: The number of components to keep.
        """
        self.n_components = n_components
        self.pca = None

    def __call__(self, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Reduces the features using incremental PCA.

        Parameters:
            features: The features to reduce with shape (num_samples, num_features).

        Returns:
            The reduced features with shape (num_samples, num_components).
        """
        LOGGER.info(
            f"Reducing features using incremental PCA with {self.n_components} "
            "components"
        )
        if self.pca is None:
            self.pca = IncrementalPCA(n_components=self.n_components)
            reduced_features = self.pca.fit_transform(features)  # type: ignore
        else:
            reduced_features = self.pca.transform(features)  # type: ignore
        LOGGER.info(f"Variance retained: {self.pca.explained_variance_ratio_.sum()}")  # type: ignore  # noqa: E501
        return reduced_features


class NoFeatureReduction(FeatureReduction):
    """No feature reduction."""

    def __call__(self, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Reduces the features using no feature reduction."""
        return features


class PCAFeatureReduction(FeatureReduction):
    """PCA feature reduction."""

    def __init__(self, n_components: int, seed: int):
        """Initialize the PCA feature reduction.

        Parameters:
            n_components: The number of components to keep.
            seed: The seed for the random number generator.
        """
        self.n_components = n_components
        self.seed = seed
        self.pca = None

    def __call__(self, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Reduces the features using PCA.

        Parameters:
            features: The features to reduce with shape (num_samples, num_features).

        Returns:
            The reduced features with shape (num_samples, num_components).
        """
        if self.pca is None:
            self.pca = PCA(n_components=self.n_components, random_state=self.seed)
            reduced_features = self.pca.fit_transform(features)  # type: ignore
        else:
            reduced_features = self.pca.transform(features)  # type: ignore
        return reduced_features


class SRPFeatureReduction(FeatureReduction):
    """Feature reduction using Sparse Random Projection."""

    def __init__(
        self,
        seed: int,
        eps: float = 0.1,
        n_components: int | Literal["auto"] = "auto",
    ):
        """Initialize the SRP feature reduction.

        Parameters:
            seed: The seed for the random number generator.
            eps: The epsilon for the Johnson-Lindenstrauss lemma.
        """
        self.seed = seed
        self.eps = eps
        self.n_components = n_components
        self.srp = None

    def __call__(self, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Reduces the features using a sparse random projection.

        Parameters:
            features: The features to reduce with shape (num_samples, num_features).

        Returns:
            The reduced features with shape (num_samples, num_projections).
        """
        if self.n_components == "auto":
            num_projections = johnson_lindenstrauss_min_dim(
                features.shape[0], eps=self.eps
            )
            # Can return a larger number of projections than feature dims,
            #  we don't want that. Reduce to at most current no features.
            num_projections = min(num_projections, features.shape[1])
        else:
            num_projections = self.n_components
        if self.srp is None:
            self.srp = SparseRandomProjection(num_projections, random_state=self.seed)
            reduced_features = self.srp.fit_transform(features)  # type: ignore
        else:
            reduced_features = self.srp.transform(features)  # type: ignore
        return reduced_features
