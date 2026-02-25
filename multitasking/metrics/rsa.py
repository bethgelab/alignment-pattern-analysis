"""Representational similarity analysis (RSA)."""

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

from .base import Metric


class RSA(Metric):
    """Representational similarity analysis (RSA)."""

    def __init__(
        self,
        distance_metric: Literal["correlation", "euclidean", "cosine"] = "correlation",
        correlation_metric: Literal["pearson", "spearman"] = "pearson",
    ):
        """Initialize the RSA metric.

        Parameters:
            distance_metric: The metric to use to compute the distance between samples.
            correlation_metric: The metric to use to compute the correlation between
                RDMs.
        """
        if distance_metric not in ["correlation", "euclidean", "cosine"]:
            raise ValueError(f"Invalid distance metric: {distance_metric}")

        if correlation_metric not in ["pearson", "spearman"]:
            raise ValueError(f"Invalid correlation metric: {correlation_metric}")

        self.correlation_metric = correlation_metric
        self.distance_metric = distance_metric

    def __call__(
        self,
        features1: npt.ArrayLike,
        features2: npt.ArrayLike,
    ) -> tuple[float, dict[str, Any]]:
        """Compute the representational similarity between two feature matrices.

        Parameters:
            features1: The first feature matrix of shape (n_samples, n_features_1).
            features2: The second feature matrix of shape (n_samples, n_features_2).

        Returns:
            score: The representational similarity score.
            details: A dictionary containing the RDMs as numpy arrays of shape
                (n_samples, n_samples). Keys are "rdm1" and "rdm2".
        """
        features1 = np.asarray(features1, dtype=np.float32)
        features2 = np.asarray(features2, dtype=np.float32)

        rdm1 = _compute_rdm(features1, self.distance_metric)
        rdm2 = _compute_rdm(features2, self.distance_metric)

        score = _compare_rdms(rdm1, rdm2, self.correlation_metric)

        details = {
            "rdm1": rdm1,
            "rdm2": rdm2,
        }

        return score, details


def _compute_rdm(
    x: npt.ArrayLike,
    metric: Literal["correlation", "euclidean", "cosine"] = "correlation",
) -> npt.NDArray[np.float32]:
    """Compute a Representational Dissimilarity Matrix (RDM).

    Parameters:
        x: Feature matrix of shape (n_samples, n_features).
        metric: The metric to use to compute the distance between samples.

    Returns:
        rdm: The RDM as a numpy array of shape (n_samples, n_samples).
    """
    if metric == "correlation" and torch.cuda.is_available():
        return _compute_rdm_torch(torch.from_numpy(x).to(device="cuda")).cpu().numpy()
    else:
        return _compute_rdm_scipy(x, metric=metric)


def _compute_rdm_scipy(
    x: npt.ArrayLike,
    metric: Literal["correlation", "euclidean", "cosine"] = "correlation",
) -> npt.NDArray[np.float32]:
    distances = pdist(x, metric=metric)
    rdm = squareform(distances)
    return rdm


def _compute_rdm_torch(
    x: torch.Tensor,
    metric: Literal["correlation"] = "correlation",
    chunk_size: int = 1000,
) -> torch.Tensor:
    if metric != "correlation":
        raise NotImplementedError(f"Metric {metric} not implemented")

    x = x - x.mean(dim=1, keepdim=True)
    x = x / x.norm(dim=1, keepdim=True)

    rdm = torch.zeros((x.shape[0], x.shape[0]))
    for i in range(0, x.shape[0], chunk_size):
        for j in range(0, x.shape[0], chunk_size):
            distance = 1 - x[i : i + chunk_size] @ x[j : j + chunk_size].T
            rdm[i : i + chunk_size, j : j + chunk_size] = distance

    return rdm


def _compare_rdms(
    rdm1: npt.NDArray[np.float32],
    rdm2: npt.NDArray[np.float32],
    correlation_metric: Literal["pearson", "spearman"] = "pearson",
) -> float:
    rdm1_triu = rdm1[np.triu_indices(rdm1.shape[0], k=1)]
    rdm2_triu = rdm2[np.triu_indices(rdm2.shape[0], k=1)]

    if correlation_metric == "pearson":
        return pearsonr(rdm1_triu, rdm2_triu)[0].item()
    elif correlation_metric == "spearman":
        return spearmanr(rdm1_triu, rdm2_triu)[0].item()
    else:
        raise ValueError(f"Invalid correlation metric: {correlation_metric}")
