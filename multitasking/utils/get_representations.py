import logging
from pathlib import Path

import numpy as np

from multitasking.datasets import BoldMoments
from multitasking.feature_extraction import FeatureBuffer, extract_model_representations
from multitasking.feature_reduction import reduce_features
from multitasking.logging_utils import memory_usage
from multitasking.utils.file_handling import config_hash

LOGGER = logging.getLogger(__name__)


def compute_model_representation(
    config: dict,
    dataset: BoldMoments,
    cache_path: Path,
    reuse: bool = False,
    split: str | None = None,
    feature_projectors: dict | None = None,
):
    """Computes the model representation.

    This function orchestrates all steps that are necessary to compute the model
    representation:

        1. Loading the stimulus dataset
        2. Loading the model
        3. Extracting the model features
        4. Reducing the model features

    The model representation may be cached after the extraction and reduction steps in
    order to avoid recomputing the same representation during development. For the final
    results, the cache should be disabled.

    Parameters:
        config: The full configuration dictionary.
        dataset: BoldMoments dataset object. Crucially, this list defines the
                order of the samples in the final representation matrix.
        cache_path: The path to the cache directory.
        reuse: Whether to reuse the cached model representation.
        split: test or train; only for bold_moments, and only for the caching path.
               the dataset determine which videos to extract features for.
        feature_projectors: A dictionary of feature projectors for each layer.
            This is required when loading predefined train and test sets; we need to
            use the same projection for both.

    Returns:
        - reduced_features: A dictionary containing the reduced model features.
        Each entry corresponds to one layer and provides a matrix of shape
        (n_samples, n_frames, n_features).
        - feature_projectors: A dictionary of feature projectors for each layer.
    """
    LOGGER.info("Computing model representation ...")

    feature_extraction_cache_path = (
        cache_path / "feature_extraction"
    )
    if split is not None:
        feature_extraction_cache_path = feature_extraction_cache_path / split
    LOGGER.info(f"Feature extraction cache path: {feature_extraction_cache_path}")

    # Determine the location of the feature reduction cache based on the relevant
    # config settings.
    feature_reduction_config_hash = config_hash(
        config, ["fmri", "dataset", "feature_extraction", "feature_reduction"]
    )
    feature_reduction_cache_path = (
        cache_path
        / "feature_reduction"
        / feature_reduction_config_hash
        / (split if split is not None else "")
        / "reduced_features.npz"
    )
    LOGGER.info(f"Feature reduction cache path: {feature_reduction_cache_path}")

    # Test whether we can reuse the cached feature extraction or reduction.
    reuse_feature_extraction = reuse and feature_extraction_cache_path.exists()
    reuse_feature_reduction = reuse and feature_reduction_cache_path.exists()
    LOGGER.info(f"Reuse extracted features: {reuse_feature_extraction}")
    LOGGER.info(f"Reuse reduced features: {reuse_feature_reduction}")
    LOGGER.info(f"reuse: {reuse}")
    LOGGER.info(f"extraction path: {feature_extraction_cache_path.exists()}")
    LOGGER.info(f"reduction path: {feature_reduction_cache_path.exists()}")

    if reuse_feature_reduction:
        # If we can reuse the cached feature reduction we load it and are done.
        LOGGER.warning(
            f"Reusing the cached reduced model representation at "
            f"{feature_reduction_cache_path}"
        )
        reduced_features = np.load(feature_reduction_cache_path)

    else:
        # Otherwise, we may be able to reuse the cached feature extraction, but have to
        # run the feature reduction in any case.
        if reuse_feature_extraction:
            LOGGER.warning(
                "Reusing the cached model representation at "
                f"{feature_extraction_cache_path}"
            )
            features = FeatureBuffer(feature_extraction_cache_path)

        else:
            LOGGER.info("Extracting model features ...")
            features = extract_model_representations(  # type: ignore
                config, dataset, feature_extraction_cache_path
            )

            # We don't need to explicitetly save the features here, since the memory
            # mapped FeatureBuffer is persisted anyway.

            LOGGER.info("Extracting model features completed.")

        LOGGER.info("Reducing model features ...")
        if feature_projectors is None:
            feature_projectors = {layer: None for layer in features.layers}

        reduced_features = dict()
        for layer in features.layers:
            N, T, H, W, C = features[layer].shape
            # The following was unindented, so only run for the last layer.
            # Probably was a bug?
            features[layer].shape = (N * T, H * W * C)
            reduced_features[layer], projector = reduce_features(
                config,
                features[layer],
                feature_projectors[layer],  # type: ignore
            )
            feature_projectors[layer] = projector
            reduced_features[layer] = (
                reduced_features[layer].reshape(N, T, -1).squeeze()
            )
            LOGGER.info(
                f"---dev-log--- squeezed red. features for {layer} shape: {
                    reduced_features[layer].shape
                    }"
            )
        feature_reduction_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(feature_reduction_cache_path, **reduced_features)
        LOGGER.info("Reducing model features completed.")
        LOGGER.info(f"Last reduced layer size: {reduced_features[layer].shape}")

    representation_info = "\n".join(
        [
            f"- {layer}: {data.shape} ({memory_usage(data)})"
            for layer, data in reduced_features.items()
        ]
    )
    LOGGER.info(
        f"Computing model representation completed.\n{representation_info}"
        )

    return reduced_features, feature_projectors


def get_flat_representation(
    nested_representation_dict: dict,
) -> dict:
    """Flatten a nested dictionary."""
    flat_representation_dict = {}
    for split, rep_dict in nested_representation_dict.items():
        for layer, rep in rep_dict.items():
            flat_representation_dict[f"{split}_{layer}"] = rep
    return flat_representation_dict

