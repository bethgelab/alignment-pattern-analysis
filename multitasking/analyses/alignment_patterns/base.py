"""Base class and utilities for alignment patterns."""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


class AlignmentPattern(ABC):
    """Base class for alignment patterns."""

    def __init__(
        self,
        metric: str,
        main_outcome: str = "score",
    ):
        """Initialize the alignment pattern."""
        self.metric = metric
        self.main_outcome = main_outcome


    @abstractmethod
    def get_alignment_pattern_df(
        self,
        df: pd.DataFrame,
        split: str,
    ) -> None:
        """Get alignment patterns from a dataframe.

        Args:
            df: DataFrame containing alignment scores with required columns.
            split: Data split to use (e.g., "train", "test").

        Returns:
            DataFrame containing alignment patterns.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    # Note: get_average_alignment_pattern is not abstract because some subclasses
    # have different signatures (e.g., PairwiseBrainBrainAlignmentPatterns)



class _SimMetricResult:
    """Result object for similarity metric, mimicking scipy.stats.pearsonr interface."""

    def __init__(self, statistic: float):
        self.statistic = statistic


def _mse_similarity(x: np.ndarray, y: np.ndarray) -> _SimMetricResult:
    """Compute Mean Squared Error between two arrays.

    Args:
        x: First array
        y: Second array

    Returns:
        _SimMetricResult object with .statistic property containing the MSE value.
    """
    mse = 1/(1+ float(np.mean((x - y) ** 2)))
    return _SimMetricResult(mse)

def _mae_similarity(x: np.ndarray, y: np.ndarray) -> _SimMetricResult:
    """Compute Mean Absolute Error between two arrays.

    Args:
        x: First array
        y: Second array

    Returns:
        _SimMetricResult object with .statistic property containing the MAE value.
    """
    mae = 1/(1+ float(np.mean(np.abs(x - y))))
    return _SimMetricResult(mae)

def _rank_correlation_similarity(x: np.ndarray, y: np.ndarray) -> _SimMetricResult:
    """Compute Rank Correlation between two arrays.

    Args:
        x: First array
        y: Second array

    Returns:
        _SimMetricResult object with .statistic property containing the rank correlation value.
    """
    rank_correlation = float(spearmanr(x, y)[0])
    return _SimMetricResult(rank_correlation)

def _cosine_similarity(x: np.ndarray, y: np.ndarray) -> _SimMetricResult:
    """Compute Cosine Similarity between two arrays.

    Args:
        x: First array
        y: Second array

    Returns:
        _SimMetricResult object with .statistic property containing the cosine similarity value.
    """
    cosine_similarity = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    return _SimMetricResult(cosine_similarity)

def _variance_explained_similarity(x: np.ndarray, y: np.ndarray) -> _SimMetricResult:
    """Compute Variance Explained between two arrays.

    Args:
        x: First array
        y: Second array

    Returns:
        _SimMetricResult object with .statistic property containing the variance explained value.
    """
    variance_explained = float(pearsonr(x, y)[0]**2)
    return _SimMetricResult(variance_explained)

def get_similarity_function(similarity_metric: str) -> Callable:
    """Get the similarity function for a given metric.

    Args:
        similarity_metric: Similarity metric name ("pearson" or "mse" or "mae" or "rank_correlation" or "cosine" or "variance_explained").

    Returns:
        Similarity function (pearsonr or _mse_similarity or _mae_similarity or _rank_correlation_similarity or _cosine_similarity).

    Raises:
        ValueError: If similarity_metric is not supported.
    """
    if similarity_metric == "pearson":
        return pearsonr
    elif similarity_metric == "mse":
        return _mse_similarity
    elif similarity_metric == "mae":
        return _mae_similarity
    elif similarity_metric == "rank_correlation":
        return _rank_correlation_similarity
    elif similarity_metric == "cosine":
        return _cosine_similarity
    elif similarity_metric == "variance_explained":
        return _variance_explained_similarity
    else:
        raise ValueError(
            f"Similarity metric must be 'pearson' or 'mse' or 'mae' or 'rank_correlation', got '{similarity_metric}'."
        )


def extract_connectivity_reference_pattern(
    roi: str,
    roi_ref_data: np.ndarray | dict[str, float],
    valid_rois: list[str],
) -> np.ndarray:
    """Extract reference alignment pattern for a ROI from connectivity data.

    Excludes the ROI from its own alignment pattern (self-connections removed).

    Args:
        roi: ROI name to extract pattern for.
        roi_ref_data: Connectivity data (dict or array).
        valid_rois: List of valid ROI names in order.

    Returns:
        Reference alignment pattern as numpy array (excluding self).

    Raises:
        ValueError: If array length doesn't match expected size.
    """
    valid_rois_excluding_self = [_r for _r in valid_rois if _r != roi]
    if len(valid_rois_excluding_self) == 0:
        raise ValueError(f"No valid ROIs remaining after excluding {roi}.")

    if isinstance(roi_ref_data, dict):
        return np.array([roi_ref_data[_roi] for _roi in valid_rois_excluding_self])

    # Array format - assume ordered according to valid_rois
    roi_ref_array = np.array(roi_ref_data)
    if roi not in valid_rois:
        if len(roi_ref_array) != len(valid_rois_excluding_self):
            raise ValueError(
                f"Length mismatch: expected {len(valid_rois_excluding_self)}, "
                f"got {len(roi_ref_array)}."
            )
        return roi_ref_array

    roi_index = valid_rois.index(roi)
    if len(roi_ref_array) != len(valid_rois):
        raise ValueError(
            f"Length mismatch: expected {len(valid_rois)}, got {len(roi_ref_array)}."
        )

    # Exclude self ROI
    return np.concatenate([
        roi_ref_array[:roi_index],
        roi_ref_array[roi_index + 1:],
    ])
