"""Wrapper for metrics from NeuroAIMetrics."""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from multitasking.metrics.anshksoni_neuroaimetrics_metrics import (
    all_metrics as METRICS_DICT,
)
from multitasking.metrics.anshksoni_neuroaimetrics_metrics import (
    is_distance_metric as IS_DISTANCE_METRIC_DICT,
)
from multitasking.metrics.base import Metric
from multitasking.metrics.evaluate_anshksoni_metric import (
    evaluate_CKA,
    evaluate_Correlation,
    evaluate_LinearPredictivity,
    evaluate_PairwiseMatching,
    evaluate_PLSreg,
    evaluate_Procrustes,
    evaluate_ReverseLinearPredictivity,
    evaluate_RSA,
    evaluate_SoftMatching,
    evaluate_SymmetricLinearPredictivity,
    evaluate_VERSA,
)

EVALUATE_FUNCTIONS = {
    "LinearPredictivity": evaluate_LinearPredictivity,
    "ReverseLinearPredictivity": evaluate_ReverseLinearPredictivity,
    "SymmetricLinearPredictivity": evaluate_SymmetricLinearPredictivity,
    "PLSreg": evaluate_PLSreg,
    "PairwiseMatching": evaluate_PairwiseMatching,
    "SoftMatching": evaluate_SoftMatching,
    "RSA": evaluate_RSA,
    "CKA": evaluate_CKA,
    "VERSA": evaluate_VERSA,
    "Procrustes": evaluate_Procrustes,
    "Correlation": evaluate_Correlation,
}

class AnshKSoniMetric(Metric):
    def __init__(self,
        metric_function_name: str,
        normalize: bool = True,
        # num_folds: int = 5,
        # seed: int = 1,
        # **kwargs
        ):
        """Initialize the metric."""
        self.metric_function_name = metric_function_name
        self.normalize = normalize
        # for key, value in kwargs.items():
        #     self.setattr(key, value)

        try:
            self.fun: Callable[..., Any] = METRICS_DICT[metric_function_name]  # type: ignore[assignment]
        except KeyError as e:
            raise ValueError(
                    f"Metric function {metric_function_name} not found. "
                    f"Available functions (not all will be valid metrics): "
                    f"{list(METRICS_DICT.keys())}"
            ) from e

        self.is_distance_metric = IS_DISTANCE_METRIC_DICT[metric_function_name]
        self.artefacts = None

        self.eval_fun = EVALUATE_FUNCTIONS[metric_function_name]


    def __call__(
        self, X: npt.ArrayLike, Y: npt.ArrayLike,
        ) -> tuple[float, dict[str, Any]]:
        """Compute the metric.

        Parameters:
            features1: The first feature matrix of shape (n_samples, n_features_1).
            features2: The second feature matrix of shape (n_samples, n_features_2).

        Returns:
            score: The similarity score.
            details: A dictionary containing:
        """
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        # * Preprocessing *
        if self.normalize:
            self.x_mean_train = np.nanmean(X, axis=0)
            self.x_sd_train = np.nanstd(X, axis=0)
            self.x_sd_train[np.isclose(self.x_sd_train, 0)] = 1
            self.y_mean_train = np.nanmean(Y, axis=0)
            self.y_sd_train = np.nanstd(Y, axis=0)
            self.y_sd_train[np.isclose(self.y_sd_train, 0)] = 1
            X = (X - self.x_mean_train) / self.x_sd_train
            Y = (Y - self.y_mean_train) / self.y_sd_train

        # * Compute score (CV implemented inside the metric) *
        X_arr = np.asarray(X)
        Y_arr = np.asarray(Y)

        if X_arr.shape[-1] == 0 or Y_arr.shape[-1] == 0:
            score = float(np.nan)  # no features to compare
            self.artefacts = None
        else:
            scores, artefacts = self.fun(X_arr, Y_arr, return_artefacts=True)
            self.artefacts = artefacts
            if isinstance(scores, list):
                score = float(np.nanmean(scores))
            else:
                score = float(scores)
            if self.is_distance_metric:
                score = -score

        return score, self.artefacts or {}


    def evaluate(self, X: npt.ArrayLike, Y: npt.ArrayLike) -> float:
        """Evaluate a previously fitted metric."""
        if self.artefacts is None:
            raise ValueError("Metric has not been fitted yet.")
        eval_score = self.eval_fun(X, Y, artefacts=self.artefacts)
        if isinstance(eval_score, list):
            eval_score = np.mean(eval_score)

        if self.is_distance_metric:
            eval_score = -eval_score

        return eval_score
