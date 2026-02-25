"""Copied (and potentially will be modified) from:.

https://github.com/anshksoni/NeuroAIMetrics/blob/main/utils/metrics.py

-- They do not provide a license, so will have to replace this with
self-written metrics in case I want to publish the code.
"""

import gc
import logging

import numpy as np
import scipy
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

np.seterr(invalid="ignore")

LOGGER = logging.getLogger(__name__)

try:
    from fastprogress.fastprogress import progress_bar  # type: ignore[import-not-found]
except ImportError:
    pass # not actually using it
try:
    import ot  # type: ignore[import-untyped, import-not-found]
except ImportError:
    LOGGER.warning("POT not found, so SoftMatching will not be available.")




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


def LinearPredictivity(X, Y, return_artefacts=False):
    # split into train and test regression
    all_corrs = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    predictor_models = []
    for _i, (train_idx, test_idx) in enumerate(kf.split(X)):
        # train regression
        predictor = RidgeCV(alphas=np.logspace(-8, 8, 17))
        predictor.fit(normalize(X[train_idx]), Y[train_idx])

        # test predictions
        y_pred = predictor.predict(normalize(X[test_idx]))
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)

        corr = many_pairwise_correlation(y_pred, Y[test_idx])
        if return_artefacts:
            predictor_models.append(predictor)
        else:
            del predictor
        gc.collect()
        all_corrs.append(corr)

    if return_artefacts:
        return all_corrs, {
            "predictor_models": predictor_models,
        }
    else:
        return all_corrs


def ReverseLinearPredictivity(Y, X, return_artefacts=False):
    # split into train and test regression
    all_corrs = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    predictor_models = []
    for _i, (train_idx, test_idx) in enumerate(kf.split(X)):
        # train regression
        predictor = RidgeCV(alphas=np.logspace(-8, 8, 17))
        predictor.fit(normalize(X[train_idx]), Y[train_idx])
        # test predictions
        y_pred = predictor.predict(normalize(X[test_idx]))
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)

        corr = many_pairwise_correlation(y_pred, Y[test_idx])
        if return_artefacts:
            predictor_models.append(predictor)
        else:
            del predictor
        gc.collect()
        all_corrs.append(corr)

    if return_artefacts:
        return all_corrs, {"predictor_models": predictor_models}
    else:
        return all_corrs


def SymmetricLinearPredictivity(X, Y, return_artefacts=False):
    """Compute the the mean of LinearPredictivity and ReverseLinearPredictivity.

    Parameters:
        X: The first feature matrix of shape (n_samples, n_features_1).
        Y: The second feature matrix of shape (n_samples, n_features_2).

    Returns:
        score: The symmetric linear predictivity score.
    """
    if return_artefacts:
        corrs_xy, arts_xy = LinearPredictivity(X, Y, return_artefacts=True)
        corrs_yx, arts_yx = LinearPredictivity(Y, X, return_artefacts=True)
        scores = (np.array(corrs_xy) +  np.array(corrs_yx)) / 2.0
        # score = (np.mean(corrs_xy) + np.mean(corrs_yx)) / 2.0
        return scores.tolist(), {
            "linear_predictivity_artefacts": arts_xy,
            "reverse_linear_predictivity_artefacts": arts_yx,
        }
    else:
        scores = np.array(LinearPredictivity(X, Y)) + np.array(LinearPredictivity(Y, X))
        return (scores / 2.0).tolist()

def SoftMatching(X, Y, itermax=400, return_artefacts=False):
    if "ot" not in globals():
        raise ImportError("POT (ot) package not imported, SoftMatching "
                            "is not available.")
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

    score = []
    soft_assignments_list = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for _i, (train_idx, test_idx) in enumerate(kf.split(X)):
        nx = X[train_idx].shape[1]
        ny = Y[train_idx].shape[1]
        dist_matrix = cdist(X[train_idx].T, Y[train_idx].T)
        soft_assignments, log = ot.emd(
            np.ones(nx) / nx,
            np.ones(ny) / ny,
            dist_matrix,
            numItermax=100000 * itermax,
            log=True,
        )
        if log["warning"] is not None:
            LOGGER.warning("SoftMatching: Did not converge, increase itermax")
            return np.nan
        if return_artefacts:
            soft_assignments_list.append(soft_assignments)
        dist_matrix = cdist(X[test_idx].T, Y[test_idx].T)
        score_before_sqrt = np.sum(soft_assignments * dist_matrix)
        # Because of floating point inaccuracies, if X == Y, this can be negative.
        # Set to zero in that case:
        if score_before_sqrt < 0:
            assert np.isclose(score_before_sqrt, 0), ("score_before_sqrt is negative "
                                        "and not close to 0: " + str(score_before_sqrt))
            score_before_sqrt = 0
        score.append(np.sqrt(score_before_sqrt) / len(test_idx))

    if return_artefacts:
        return score, {"soft_assignments": soft_assignments_list}
    else:
        return score


def PairwiseMatching(X, Y, return_artefacts=False):
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    all_corrs = []
    feature_indices_list = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for _i, (train_idx, test_idx) in enumerate(kf.split(X)):
        m_codes_ = [normalize(X[train_idx]), normalize(X[test_idx])]
        gts_ = [Y[train_idx], Y[test_idx]]

        m_codes, gts = [], []
        for code in m_codes_:
            m_codes.append(torch.tensor(code))
        for code in gts_:
            gts.append(torch.tensor(code))

        A = m_codes[0]  # [:,:dims]

        B = gts[0]
        N = B.shape[0]

        # Store columnw-wise in A and B, as they would be used at few places
        sA = A.sum(0)
        sB = B.sum(0)

        # Basically there are four parts in the formula.
        # We would compute them one-by-one
        p1 = N * torch.einsum("ij,ik->kj", A, B)
        p2 = sA * sB[:, None]
        p3 = N * ((B**2).sum(0)) - (sB**2)
        p4 = N * ((A**2).sum(0)) - (sA**2)

        pcorr = (p1 - p2) / torch.sqrt(p4 * p3[:, None])

        # print(pcorr)
        pcorr[torch.isnan(pcorr)] = 0

        test_mat = m_codes[1]
        test_mat[torch.isinf(test_mat)] = 0
        indices = torch.argmax(pcorr, 1)
        if return_artefacts:
            feature_indices_list.append(indices.cpu().detach().numpy())
        test_mat = test_mat.cpu().detach().numpy()
        gt = gts[1].cpu().detach().numpy()

        corr = many_pairwise_correlation(gt, test_mat[:, indices])
        gc.collect()
        all_corrs.append(corr)
    if return_artefacts:
        return all_corrs, {"feature_indices": feature_indices_list}
    else:
        return all_corrs


def RSA(X, Y, correlation_metric="kendalltau", return_artefacts=False):
    temp = 1 - np.corrcoef(Y)
    temp2 = 1 - np.corrcoef(X)
    temp = np.array(temp[np.triu_indices(temp.shape[0], k=1)])
    temp2 = np.array(temp2[np.triu_indices(temp2.shape[0], k=1)])
    # temp = np.array(temp.flatten())
    # temp2 = np.array(temp2.flatten())
    if correlation_metric == "kendalltau":
        result = [kendalltau(temp, temp2, nan_policy="raise")[0]]
    elif correlation_metric == "pearson":
        result = [pearsonr(temp, temp2, )[0]]
    elif correlation_metric == "spearman":
        result = [spearmanr(temp, temp2, nan_policy="raise")[0]]
    else:
        raise ValueError(f"Invalid correlation metric: {correlation_metric}")
    if return_artefacts:
        return result, {"correlation_metric": correlation_metric}
    else:
        return result


def CKA(X, Y, vectorized=True, return_artefacts=False):
    # Implements linear CKA as in Kornblith et al. (2019)
    # Center X and Y

    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    gc.collect()
    YTX = Y.T.dot(X)

    def traceATA(A): # OK: This seems extremely inefficient, vectorize?
        num_rows, _ = A.shape
        out = 0
        for i in progress_bar(range(num_rows)):
            for j in range(num_rows):
                out += np.sum(A[i] * A[j]) ** 2
        return out

    def traceATA_vectorized(A):
        return np.sum((A@A.T)**2)

    # assert np.allclose(traceATA(X), traceATA_vectorized(X))
    # result_original = [(YTX**2).sum() / np.sqrt(traceATA(X) * traceATA(Y))]
    # result_vectorized = [(YTX**2).sum() / np.sqrt(traceATA_vectorized(X) \
    #                                               * traceATA_vectorized(Y))]
    # assert np.allclose(result_original, result_vectorized)
    if vectorized:
        result = [(YTX**2).sum() / np.sqrt(traceATA_vectorized(X) \
                                            * traceATA_vectorized(Y))]
    else:
        result = [(YTX**2).sum() / np.sqrt(traceATA(X) * traceATA(Y))]
    if return_artefacts:
        return result, {}
    else:
        return result


def VERSA(X, Y, return_artefacts=False):
    all_corrs = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    predictor_models = []
    for _i, (train_idx, test_idx) in enumerate(kf.split(X)):
        # train regression
        predictor = RidgeCV(alphas=np.logspace(-8, 8, 17))
        predictor.fit(normalize(X[train_idx]), Y[train_idx])

        # test predictions
        y_pred = predictor.predict(normalize(X[test_idx]))
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        model_rdm = 1 - np.corrcoef(y_pred)
        Y_rdms = 1 - np.corrcoef(Y[test_idx])
        temp = np.array(model_rdm[np.triu_indices(model_rdm.shape[0], k=1)])
        temp2 = np.array(Y_rdms[np.triu_indices(Y_rdms.shape[0], k=1)])

        all_corrs.append(kendalltau(temp, temp2)[0])
        if return_artefacts:
            predictor_models.append(predictor)
        else:
            del predictor
        gc.collect()

    if return_artefacts:
        return all_corrs, {"predictor_models": predictor_models}
    else:
        return all_corrs


def Procrustes(
    X,
    Y,
    alpha=1,
    feature_size_adjustment="pca",
    divide_by_norm=True,
    return_train_score=False,
    return_artefacts=False,
):
    """Returns the arccos of the procrustes similarity between X and Y.

    This means that higher is more distant, and the values will be radians.

    :param alpha: Set it to 1 to NOT use any whitening. Set it to 0 to
                  whiten completely. Anything in between will partially
                  whiten.
     :param divide_by_norm: Divide the matrices by the frobenius
                       norm to get comparable scales between train and test
                       data, as well as X and Y. (Todo: Is that already done
                       in the code below without divide_by_norm=True?)
    """
    if feature_size_adjustment not in ["pca", "zero-pad"]:
        raise ValueError(
            f"Invalid feature size adjustment method: {feature_size_adjustment}"
        )

    if return_artefacts and alpha != 1:
        raise ValueError(
            "Procrustes: return_artefacts=True requires alpha=1 (no whitening)."
        )

    def whiten(X, alpha=alpha, preserve_variance=True, eigval_tol=1e-7):
        if alpha > (1 - eigval_tol):
            return X, np.eye(X.shape[1])

        # Compute eigendecomposition of covariance matrix
        lam, V = np.linalg.eigh(X.T @ X)
        lam = np.maximum(lam, eigval_tol)

        d = alpha + (1 - alpha) * lam ** (-1 / 2)

        # Rescale the whitening matrix.
        if preserve_variance:
            new_var = np.sum(
                (alpha**2) * lam
                + 2 * alpha * (1 - alpha) * (lam**0.5)
                + ((1 - alpha) ** 2) * np.ones_like(lam)
            )

            # Now re-scale d so that the variance of (X @ Z)
            # will equal the original variance of X.
            d *= np.sqrt(np.sum(lam) / new_var)

        # Form (partial) whitening matrix.
        Z = (V * d[None, :]) @ V.T

        return X @ Z, Z

    def partial_fit(X, alpha=alpha):
        mx = np.mean(X, axis=0)
        Xw, Zx = whiten(X - mx[None, :], alpha, preserve_variance=True)
        if divide_by_norm:
            Xw = Xw / np.linalg.norm(Xw)
        return (X, mx, Xw, Zx)

    def compute_distance(X, Y, X_test, Y_test):
        cache_X = partial_fit(X)
        cache_Y = partial_fit(Y)
        # Extract whitened representations and caches
        X_train, mx_, Xw, Zx = cache_X
        Y_train, my_, Yw, Zy = cache_Y
        X, Y, X_test, Y_test, R = finalize_fit(cache_X, cache_Y, X_test, Y_test)

        dist_test = angular_distance(X_test, Y_test)
        if return_train_score:
            dist_train = angular_distance(X, Y)
            return dist_train, dist_test, R, mx_, my_
        else:
            return dist_test, R, mx_, my_

    def angular_distance(X, Y):
        normalizer = np.linalg.norm(X.ravel()) * np.linalg.norm(Y.ravel())
        corr = np.dot(X.ravel(), Y.ravel()) / normalizer
        return np.arccos(np.clip(corr, -1.0, 1.0))

    def finalize_fit(cache_X, cache_Y, X_test, Y_test):
        # Extract whitened representations.
        X, mx_, Xw, Zx = cache_X
        Y, my_, Yw, Zy = cache_Y
        # Fit optimal rotational alignment.
        U, _, Vt = scipy.linalg.svd(Xw.T @ Yw, lapack_driver="gesvd")
        Wx_ = Zx @ U
        Wy_ = Zy @ Vt.T

        X_test = X_test - mx_[None, :]
        Y_test = Y_test - my_[None, :]

        if divide_by_norm:
            X_test = X_test / np.linalg.norm(X_test)
            Y_test = Y_test / np.linalg.norm(Y_test)

        return (
            (X - mx_[None, :]) @ Wx_,
            (Y - my_[None, :]) @ Wy_,
            (X_test) @ Wx_,
            (Y_test) @ Wy_,
            Wx_ @ Wy_.T, # rotation matrix; = U @ Vt if no whitening
        )

    pca_X = None
    pca_Y = None
    if feature_size_adjustment == "pca":
        n = min(X.shape[-1], Y.shape[-1], Y.shape[0])
        if X.shape[-1] != n:
            pca_X = PCA(n, random_state=42).fit(X)
            X = pca_X.transform(X)
        if Y.shape[-1] != n:
            pca_Y = PCA(n, random_state=42).fit(Y)
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
    if X.shape[0] <= X.shape[1]:
        LOGGER.warning(
            f"Procrustes: number of samples smaller than num dimensions, "
            f"procrustes likely to overfit: ({X.shape[0]} <= {X.shape[1]})"
        )

    all_dists_test = []
    all_dists_train = []
    procrustes_transforms = []
    means_X = []
    means_Y = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for _i, (train_idx, test_idx) in enumerate(kf.split(X)):
        results = compute_distance(X[train_idx], Y[train_idx], X[test_idx], Y[test_idx])
        if return_train_score:
            D_train, D_test, R, mx_, my_ = results
            all_dists_train.append(D_train.sum())
            all_dists_test.append(D_test.sum())
        else:
            D_test, R, mx_, my_ = results
            all_dists_test.append(D_test.sum())
        if return_artefacts:
            procrustes_transforms.append(R)
            means_X.append(mx_)
            means_Y.append(my_)

    # * Return results
    if return_artefacts:
        details = {"procrustes_transforms": procrustes_transforms,
                    "pca_X": pca_X,
                    "pca_Y": pca_Y,
                    "alpha": alpha,
                    "feature_size_adjustment": feature_size_adjustment,
                    "divide_by_norm": divide_by_norm,
                    "means_X": means_X,
                    "means_Y": means_Y}
        if return_train_score:
            return all_dists_train, all_dists_test, details
        else:
            return all_dists_test, details
    else:
        if return_train_score:
            return all_dists_train, all_dists_test
        else:
            return all_dists_test


def PLSreg(X, Y, return_artefacts=False):
    all_corrs = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    predictor_models = []
    for _i, (train_idx, test_idx) in enumerate(kf.split(X)):
        n_components = np.min([25, X.shape[1]])
        predictor = PLSRegression(n_components=n_components)
        predictor.fit(normalize(X[train_idx]), Y[train_idx])

        y_pred = predictor.predict(normalize(X[test_idx]))
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)

        corr = many_pairwise_correlation(y_pred, Y[test_idx])
        if return_artefacts:
            predictor_models.append(predictor)
        else:
            del predictor
        gc.collect()
        all_corrs.append(corr)

    if return_artefacts:
        return all_corrs, {"predictor_models": predictor_models}
    else:
        return all_corrs

def Correlation(X, Y, return_artefacts=False):
    """Correlate X[:, i] with Y[:, i] for all i up to min(d_X, d_Y).

    Then take the mean over i.
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    # normalize each feature to variance 1
    X = X / np.std(X, axis=0, keepdims=True)
    Y = Y / np.std(Y, axis=0, keepdims=True)
    min_dim = min(X.shape[1], Y.shape[1])
    assert X.shape[0] == Y.shape[0], "cannot correlate vectors of different length"
    assert X.shape[0] >= 2, "cannot correlate single numbers"
    corr = X.T @ Y / (X.shape[0] - 1)
    # take the mean
    result = np.trace(corr) / min_dim
    if return_artefacts:
        return result, {}
    else:
        return result

all_metrics = {
    "LinearPredictivity": LinearPredictivity,
    "ReverseLinearPredictivity": ReverseLinearPredictivity,
    "SymmetricLinearPredictivity": SymmetricLinearPredictivity,
    "PLSreg": PLSreg,
    "PairwiseMatching": PairwiseMatching,
    "SoftMatching": SoftMatching,
    "RSA": RSA,
    "VERSA": VERSA,
    "CKA": CKA,
    "Procrustes": Procrustes,
    "Correlation": Correlation,
}
is_distance_metric = {
    "LinearPredictivity": False,
    "ReverseLinearPredictivity": False,
    "SymmetricLinearPredictivity": False,
    "PLSreg": False,
    "PairwiseMatching": False,
    "SoftMatching": True,
    "RSA": False,
    "RSAPearson": False,
    "RSASpearman": False,
    "VERSA": False,
    "CKA": False,
    "Procrustes": True,
    "Procrustes_zero_pad": True,
    "Procrustes_do_whiten": True,
    "Procrustes_do_whiten_zero_pad": True,
    "Correlation": False,
}
modified_ashksoni_metrics = {
    "Procrustes_zero_pad": lambda X, Y, **kwargs: Procrustes(
        X, Y, feature_size_adjustment="zero-pad", **kwargs
    ),
    "Procrustes_do_whiten": lambda X, Y, **kwargs: Procrustes(X, Y, alpha=0, **kwargs),
    "Procrustes_do_whiten_zero_pad": lambda X, Y, **kwargs: Procrustes(
        X, Y, alpha=0, feature_size_adjustment="zero-pad", **kwargs
    ),
    "RSAPearson": lambda X, Y, **kwargs: RSA(X, Y, correlation_metric="pearson"),
    "RSASpearman": lambda X, Y, **kwargs: RSA(X, Y, correlation_metric="spearman"),
}

