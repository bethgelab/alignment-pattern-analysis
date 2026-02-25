"""Copied (and potentially will be modified) from:.

https://github.com/anshksoni/NeuroAIMetrics/blob/main/utils/metrics.py.

-- They do not provide a license, so will have to replace this with
self-written metrics in case I want to publish the code.
"""

import gc
import logging

import numpy as np
from scipy.stats import kendalltau

try:
    import metrics_benchmark.metrics.anshksoni_neuroaimetrics_metrics as AKSMetrics  # type: ignore[import-untyped, import-not-found]
except ImportError:
    import multitasking.metrics.anshksoni_neuroaimetrics_metrics as AKSMetrics


np.seterr(invalid="ignore")

LOGGER = logging.getLogger(__name__)


def many_pairwise_correlation(A, B):
    if len(A.shape) == 1:
        A = A.reshape(-1, 1)
    if len(B.shape) == 1:
        B = B.reshape(-1, 1)
    memsize = 10000
    corr = []
    lens = []
    for i in range(0, A.shape[1], memsize):
        model_corrs = pairwise_correlation(A[:, i : i + memsize], B[:, i : i + memsize])
        corr.append(np.nanmean(np.diag(model_corrs)))
        lens.append(A[:, i : i + memsize].shape[1])
    corr = np.sum(np.multiply(np.array(corr), np.array(lens))) / np.sum(lens)
    gc.collect()
    return corr


def cdist(X, Y):
    norms_X = np.sum(X**2, axis=1)
    norms_Y = np.sum(Y**2, axis=1)
    cross_term = np.dot(X, Y.T)
    dist_matrix = norms_X[:, np.newaxis] + norms_Y[np.newaxis, :] - 2 * cross_term
    return dist_matrix


def cossim(x, y):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    x_norm[x_norm == 0] = 1e-8
    y_norm[y_norm == 0] = 1e-8
    x_normalized = x / x_norm
    y_normalized = y / y_norm
    cos_sim = x_normalized @ y_normalized.T
    return cos_sim


def pairwise_correlation(A, B):
    am = A - np.mean(A, axis=0, keepdims=True)
    bm = B - np.mean(B, axis=0, keepdims=True)
    return (
        am.T
        @ bm
        / (
            np.sqrt(np.sum(am**2, axis=0, keepdims=True)).T
            * np.sqrt(np.sum(bm**2, axis=0, keepdims=True))
        )
    )


def normalize(X):
    mean = np.nanmean(X, 0)  # training set

    stddev = np.nanstd(X, 0)  # training set
    X_zm = X - mean
    X_zm_unit = X_zm / stddev
    X_zm_unit[np.isnan(X_zm_unit)] = 0
    return X_zm_unit


def evaluate_LinearPredictivity(X, Y, artefacts):
    # split into train and test regression
    all_corrs = []
    predictor_models = artefacts["predictor_models"]
    for predictor in predictor_models:
        y_pred = predictor.predict(normalize(X))
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)

        corr = many_pairwise_correlation(y_pred, Y)
        all_corrs.append(corr)

    return all_corrs


def evaluate_ReverseLinearPredictivity(Y, X, artefacts):
    # split into train and test regression
    all_corrs = []
    predictor_models = artefacts["predictor_models"]
    for predictor in predictor_models:
        y_pred = predictor.predict(normalize(X))
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        corr = many_pairwise_correlation(y_pred, Y)
        all_corrs.append(corr)

    return all_corrs


def evaluate_SymmetricLinearPredictivity(X, Y, artefacts):
    """Compute the the mean of LinearPredictivity and ReverseLinearPredictivity.

    Parameters:
        X: The first feature matrix of shape (n_samples, n_features_1).
        Y: The second feature matrix of shape (n_samples, n_features_2).

    Returns:
        score: The symmetric linear predictivity score.
    """
    scores_xy = evaluate_LinearPredictivity(
        X, Y, artefacts["linear_predictivity_artefacts"])
    scores_yx = evaluate_LinearPredictivity(
        Y, X, artefacts["reverse_linear_predictivity_artefacts"])
    return ((np.array(scores_xy) + np.array(scores_yx)) / 2.0).tolist()

def evaluate_SoftMatching(X, Y, artefacts):
    soft_assignments_list = artefacts["soft_assignments"]
    rem = []
    for i in range(X.shape[1]):
        if np.all(X[:, i] == 0):
            rem.append(i)
    X = np.delete(X, rem, axis=1)
    rem = []
    for i in range(Y.shape[1]):
        if np.all(Y[:, i] == 0):
            rem.append(i)
    Y = np.delete(Y, rem, axis=1)

    scores = []
    for _i, soft_assignments in enumerate(soft_assignments_list):
        dist_matrix = cdist(X.T, Y.T)
        score_before_sqrt = np.sum(soft_assignments * dist_matrix)
        # Because of floating point inaccuracies, if X == Y, this can be negative.
        # Set to zero in that case:
        if score_before_sqrt < 0:
            assert np.isclose(score_before_sqrt, 0), ("score_before_sqrt is negative "
                                        "and not close to 0: " + str(score_before_sqrt))
            score_before_sqrt = 0
        scores.append(np.sqrt(score_before_sqrt) / X.shape[0])

    return scores


def evaluate_PairwiseMatching(X, Y, artefacts):
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    all_corrs = []
    feature_indices_list = artefacts["feature_indices"]
    X = normalize(X)
    # \redundant / weird that it's only normalizing X, but same as in original function
    for feature_indices in (feature_indices_list):
        corr = many_pairwise_correlation(X[:, feature_indices], Y)
        all_corrs.append(corr)
    return all_corrs


def evaluate_RSA(X, Y, artefacts=None):
    if artefacts is None:
        artefacts = {"correlation_metric": "kendalltau"}
    return AKSMetrics.RSA(X, Y,
                        correlation_metric=artefacts["correlation_metric"],
                        return_artefacts=False)


def evaluate_CKA(X, Y, artefacts=None):
    # Implements linear CKA as in Kornblith et al. (2019)
    # Center X and Y
    return AKSMetrics.CKA(X, Y, vectorized=True, return_artefacts=False)


def evaluate_VERSA(X, Y, artefacts):
    predictor_models = artefacts["predictor_models"]
    all_corrs = []
    for predictor in predictor_models:
        y_pred = predictor.predict(normalize(X))
        model_rdm = 1 - np.corrcoef(y_pred)
        Y_rdms = 1 - np.corrcoef(Y)
        temp = np.array(model_rdm[np.triu_indices(model_rdm.shape[0], k=1)])
        temp2 = np.array(Y_rdms[np.triu_indices(Y_rdms.shape[0], k=1)])
        all_corrs.append(kendalltau(temp, temp2)[0])

    return all_corrs


def evaluate_Procrustes(
    X,
    Y,
    artefacts
):
    """Returns the arccos of the procrustes similarity between X and Y.

    This means that higher is more distant, and the values will be radians.

    Does not whiten.

    :param alpha: Set it to 1 to NOT use any whitening. Set it to 0 to
                  whiten completely. Anything in between will partially
                  whiten.
     :param divide_by_norm: Divide the matrices by the frobenius
                       norm to get comparable scales between train and test
                       data, as well as X and Y. (Todo: Is that already done
                       in the code below without divide_by_norm=True?)
    """
    alpha = artefacts["alpha"]
    if alpha != 1:
        raise ValueError("Procrustes: alpha != 1 is not supported for evaluation.")
    feature_size_adjustment = artefacts["feature_size_adjustment"]
    divide_by_norm = artefacts["divide_by_norm"]
    means_X = artefacts["means_X"]
    means_Y = artefacts["means_Y"]
    pca_X = artefacts["pca_X"]
    pca_Y = artefacts["pca_Y"]
    procrustes_transforms = artefacts["procrustes_transforms"]

    def partial_fit(X, mx_):
        X = X - mx_[None, :]
        if divide_by_norm:
            X = X / np.linalg.norm(X)
        return X

    def compute_distance(X, Y, mx_, my_, R):
        X = partial_fit(X, mx_)
        Y = partial_fit(Y, my_)

        X = X @ R

        dist_test = angular_distance(X, Y)
        return dist_test

    def angular_distance(X, Y):
        normalizer = np.linalg.norm(X.ravel()) * np.linalg.norm(Y.ravel())
        corr = np.dot(X.ravel(), Y.ravel()) / normalizer
        return np.arccos(np.clip(corr, -1.0, 1.0))

    if feature_size_adjustment == "pca":
        n = min(X.shape[-1], Y.shape[-1])
        # \ cannot use Y.shape[0] as test set might have different size
        if pca_X is not None:
            n = min(n, pca_X.n_components_)
        if pca_Y is not None:
            n = min(n, pca_Y.n_components_)
        if X.shape[-1] != n:
            X = pca_X.transform(X)
        if Y.shape[-1] != n:
            Y = pca_Y.transform(Y)
    elif feature_size_adjustment == "zero-pad":
        n = max(X.shape[-1], Y.shape[-1])
        if X.shape[-1] != n:
            X = np.pad(X, ((0, 0), (0, n - X.shape[-1])), mode="constant")
        if Y.shape[-1] != n:
            Y = np.pad(Y, ((0, 0), (0, n - Y.shape[-1])), mode="constant")
    else:
        raise ValueError(
            f"Invalid feature size adjustment method: {feature_size_adjustment}"
        )

    all_dists = []
    for i, R in enumerate(procrustes_transforms):
        results = compute_distance(X, Y, means_X[i], means_Y[i], R)
        all_dists.append(results)

    return all_dists


def evaluate_PLSreg(X, Y, artefacts):
    models = artefacts["predictor_models"]
    all_corrs = []
    X = normalize(X)
    for model in models:
        y_pred = model.predict(X)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        corr = many_pairwise_correlation(y_pred, Y)
        all_corrs.append(corr)
    return all_corrs


def evaluate_Correlation(X, Y, artefacts=None):
    return AKSMetrics.Correlation(X, Y, return_artefacts=False)
