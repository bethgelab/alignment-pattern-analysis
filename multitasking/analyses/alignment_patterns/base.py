"""Base class and utilities for alignment patterns."""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


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
            metric: How to compute similarity between alignment patterns. Currently only "pearson" is supported.
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



def get_similarity_function(similarity_metric: str) -> Callable:
    """Get the similarity function for a given metric.

    Args:
        similarity_metric: Similarity metric name. Only ``"pearson"`` is
            supported.

    Returns:
        Similarity function (``scipy.stats.pearsonr``).

    Raises:
        ValueError: If similarity_metric is not supported.
    """
    if similarity_metric == "pearson":
        return pearsonr
    else:
        raise ValueError(
            "Unsupported similarity metric "
            f"'{similarity_metric}'. Supported metrics are 'pearson'."
        )


def vectorized_similarity_mean(
    ref: np.ndarray,
    targets: np.ndarray,
    metric: str,
) -> float:
    """Compute mean Pearson correlation between ``ref`` and rows of ``targets``.

    Vectorized implementation for Pearson correlation. Use this instead of
    looping over targets with :func:`get_similarity_function` for better
    performance.

    Args:
        ref: Reference pattern, shape ``(n_features,)``.
        targets: Target patterns, shape ``(n_targets, n_features)``.
        metric: Similarity metric name.

    Returns:
        Mean similarity between ref and each row of targets.
    """
    if targets.size == 0:
        return np.nan
    targets = np.atleast_2d(targets)
    if metric != "pearson":
        raise ValueError(f"Unknown metric: {metric}. Supported metrics are 'pearson'.")
    stacked = np.vstack([ref, targets])
    corrs = np.corrcoef(stacked)[0, 1:]
    return float(np.mean(corrs))
