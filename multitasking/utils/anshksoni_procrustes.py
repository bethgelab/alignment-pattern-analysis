import numpy as np
import scipy
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

"""
Taken from here: https://github.com/ankhksoni/NeuroAIMetrics
"""


def Procrustes(X, Y, alpha=1, return_similarity=False):
    """Return the arccos of the procrustes similarity between X and Y.

    This means that higher is more distant, and the values will be radians.
    """

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
        return (X, mx, Xw, Zx)

    def compute_distance(X, Y, X_test, Y_test, return_similarity=False):
        X, Y, X_test, Y_test = finalize_fit(
            partial_fit(X), partial_fit(Y), X_test, Y_test
        )

        if return_similarity:
            dist_test = similarity(X_test, Y_test)
            dist_train = similarity(X, Y)
        else:
            dist_test = angular_distance(X_test, Y_test)
            dist_train = angular_distance(X, Y)
        return dist_test, dist_train

    def angular_distance(X, Y):
        normalizer = np.linalg.norm(X.ravel()) * np.linalg.norm(Y.ravel())
        corr = np.dot(X.ravel(), Y.ravel()) / normalizer
        return np.arccos(np.clip(corr, -1.0, 1.0))

    def similarity(X, Y):
        normalizer = np.linalg.norm(X.ravel()) * np.linalg.norm(Y.ravel())
        sim = np.dot(X.ravel(), Y.ravel()) / normalizer
        return sim

    def finalize_fit(cache_X, cache_Y, X_test, Y_test):
        # Extract whitened representations.
        X, mx_, Xw, Zx = cache_X
        Y, my_, Yw, Zy = cache_Y
        # Fit optimal rotational alignment.
        U, _, Vt = scipy.linalg.svd(Xw.T @ Yw, lapack_driver="gesvd")
        Wx_ = Zx @ U
        Wy_ = Zy @ Vt.T

        return (
            (X - mx_[None, :]) @ Wx_,
            (Y - my_[None, :]) @ Wy_,
            (X_test - mx_[None, :]) @ Wx_,
            (Y_test - my_[None, :]) @ Wy_,
        )

    n = min(X.shape[-1], Y.shape[-1], Y.shape[0])

    if X.shape[-1] != n:
        pca = PCA(n, random_state=42)
        X = pca.fit_transform(X)
    if Y.shape[-1] != n:
        pca = PCA(n, random_state=42)
        Y = pca.fit_transform(Y)

    all_dists_test = []
    all_dists_train = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for _, (train_idx, test_idx) in enumerate(kf.split(X)):
        D_test, D_train = compute_distance(
            X[train_idx],
            Y[train_idx],
            X[test_idx],
            Y[test_idx],
            return_similarity=return_similarity,
        )
        all_dists_test.append(D_test.sum())
        all_dists_train.append(D_train.sum())
    return np.mean(all_dists_train), np.mean(all_dists_test)
