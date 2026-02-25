"""Base class for alignment metrics."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt


class Metric(ABC):
    """Base class for alignment metrics."""

    @abstractmethod
    def __call__(
        self,
        features1: npt.NDArray[np.float32],
        features2: npt.NDArray[np.float32],
    ) -> tuple[float, dict[str, Any]]:
        """Compute the representational alignment between two feature matrices.

        If optional keyword arguments test_features{1, 2} are defined, train
        the metric parameters on the first two features and evaluate and report
        score for test features. Currently only relevant for linear predictivity

        Parameters:
            features1: First feature matrix with shape (n_samples, n_features_1).
            features2: Second feature matrix with shape
                (n_samples, n_features_2).
            test_features1: First feature matrix with shape
                (n_samples, n_features_1), test set.
            test_features2: Second feature matrix with shape
                (n_samples, n_features_2), test set.

        Returns:
            Tuple containing the score and details.
        """
        pass
