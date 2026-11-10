"""Base class and utilities for alignment patterns."""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

#: Similarity metrics the alignment-pattern analysis understands.
#:
#: Every metric here is oriented so that **larger is more similar**, which the
#: whole downstream pipeline relies on: ``aggregate_results`` picks the best
#: model as the maximum of ``aps_score_mean``, the equivalence analysis tests
#: ``aps_score_mean >= best_aps_ci_lower``, and figure 1 draws the brain-brain
#: similarity as the right-hand ceiling of the scatter panel. ``"mse"`` is
#: therefore reported *negated* -- see :func:`negative_mean_squared_error`.
SIMILARITY_METRICS = ("pearson", "mse")


class AlignmentPattern(ABC):
    """Abstract base class for alignment pattern computations.

    Subclasses define how to extract and aggregate alignment patterns from
    tabular alignment results (e.g., model–brain or brain–brain scores).
    """

    def __init__(
        self,
        metric: str,
        main_outcome: str = "score",
    ):
        """Initialize an alignment pattern object.

        Args:
            metric: How to compute similarity between alignment patterns. One of
                :data:`SIMILARITY_METRICS`.
            main_outcome: Column of the input dataframe to treat as the primary score.
        """
        self.metric = metric
        self.main_outcome = main_outcome


    @abstractmethod
    def get_alignment_pattern_df(
        self,
        df: pd.DataFrame,
        split: str,
    ) -> None:
        """Extract and store alignment patterns from a dataframe.

        Args:
            df: DataFrame containing alignment scores with required columns.
            split: Data split to use (e.g., "train", "test").

        Returns:
            None. Implementations are expected to populate ``self.alignment_pattern_df``.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    # Note: get_average_alignment_pattern is not abstract because some subclasses
    # have different signatures (e.g., PairwiseBrainBrainAlignmentPatterns)



def negative_mean_squared_error(x: np.ndarray, y: np.ndarray) -> float:
    """Returns the negated mean squared error between two alignment patterns.

    The sign flip is deliberate and is what makes MSE usable as a *similarity*:
    the rest of the pipeline assumes larger means more similar (best model =
    argmax, equivalence = at or above the best model's CI lower bound,
    brain-brain similarity = the ceiling). The returned value is therefore
    ``<= 0``, with ``0`` meaning two identical patterns.

    Unlike Pearson correlation, this is sensitive to scale and offset: two
    patterns with the same shape but different magnitudes are *not* equivalent
    under this metric. That is the point of offering it alongside ``pearson``,
    but it also means the numbers are only comparable within one
    ``(metric, main_outcome)`` combination.

    Args:
        x: First pattern, shape ``(n_features,)``.
        y: Second pattern, shape ``(n_features,)``.

    Returns:
        ``-mean((x - y) ** 2)`` as a float.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(-np.mean((x - y) ** 2))


def get_similarity_function(similarity_metric: str) -> Callable:
    """Get the similarity function for a given metric.

    Args:
        similarity_metric: Similarity metric name, one of
            :data:`SIMILARITY_METRICS`.

    Returns:
        A function of two patterns: :func:`scipy.stats.pearsonr` for
        ``"pearson"``, :func:`negative_mean_squared_error` for ``"mse"``.

    Raises:
        ValueError: If similarity_metric is not supported.
    """
    if similarity_metric == "pearson":
        return pearsonr
    elif similarity_metric == "mse":
        return negative_mean_squared_error
    else:
        raise ValueError(
            "Unsupported similarity metric "
            f"'{similarity_metric}'. Supported metrics are "
            f"{', '.join(repr(m) for m in SIMILARITY_METRICS)}."
        )


def vectorized_similarity_mean(
    ref: np.ndarray,
    targets: np.ndarray,
    metric: str,
) -> float:
    """Compute the mean similarity between ``ref`` and each row of ``targets``.

    Vectorized implementation. Use this instead of looping over targets with
    :func:`get_similarity_function` for better performance.

    Args:
        ref: Reference pattern, shape ``(n_features,)``.
        targets: Target patterns, shape ``(n_targets, n_features)``.
        metric: Similarity metric name, one of :data:`SIMILARITY_METRICS`.
            ``"mse"`` is negated, so that larger is more similar for every
            metric -- see :func:`negative_mean_squared_error`.

    Returns:
        Mean similarity between ref and each row of targets.

    Raises:
        ValueError: If metric is not supported.
    """
    if targets.size == 0:
        return np.nan
    targets = np.atleast_2d(targets)
    if metric == "pearson":
        stacked = np.vstack([ref, targets])
        corrs = np.corrcoef(stacked)[0, 1:]
        return float(np.mean(corrs))
    elif metric == "mse":
        ref = np.asarray(ref, dtype=float)
        targets = np.asarray(targets, dtype=float)
        return float(-np.mean(np.mean((targets - ref) ** 2, axis=1)))
    else:
        raise ValueError(
            f"Unknown metric: {metric}. Supported metrics are "
            f"{', '.join(repr(m) for m in SIMILARITY_METRICS)}."
        )
