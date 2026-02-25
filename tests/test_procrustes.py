import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from multitasking.utils.procrustes import OrthogonalProcrustes

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def test_basic_functionality():
    # Test with identical matrices
    X = np.random.randn(100, 20)
    proc = OrthogonalProcrustes()
    score = proc.fit(X, X)
    assert np.isclose(score, 1.0)  # Frobenius score should be 1 for identical matrices

    # Test with rotated matrix
    Q = np.linalg.qr(np.random.randn(20, 20))[0]  # Random orthogonal matrix
    Y = X @ Q
    score = proc.fit(X, Y)
    assert np.isclose(score, 1.0)  # Should recover rotation perfectly

    # Test with different data, same Q
    X2 = np.random.randn(100, 20)
    Y2 = X2 @ Q
    score = proc.score(X2, Y2)
    assert np.isclose(score, 1.0)  # Should recover rotation perfectly

    # IDEA test this with dimensionality reduction

    # test different N dimensions but same data
    Y = np.random.randn(100, 20)
    X2 = np.concatenate((X, X, X), axis=0)
    Y2 = np.concatenate((Y, Y, Y), axis=0)
    proc = OrthogonalProcrustes()
    score1 = proc.fit(X, Y)
    score1b = proc.score(X, Y)
    assert np.isclose(score1, score1b), f"score1: {score1}, score1b: {score1b} "

    proc = OrthogonalProcrustes()
    score2 = proc.fit(X2, Y2)
    score2b = proc.score(X2, Y2)
    assert np.isclose(score2, score2b), f"score2: {score2}, score2b: {score2b} "

    assert np.isclose(score1, score2), (
        f"score1: {score1}, score2: {score2} - should be close "
        "since data is just repeated"
    )


def test_score_vs_fit():
    for _ in range(3):
        X = np.random.rand(100, 40)
        Y = np.random.randn(100, 30)

        # change distribution of X and Y to be more varied
        # pick one of three distributions randomly
        class Distrib:
            def __init__(self, distr):
                self.distr = distr

            def __call__(self, n, d):
                try:
                    arr = self.distr(size=n * d)
                    return arr.reshape(n, d)
                except TypeError:  # randn
                    return self.distr(n, d)

        distributions = [
            np.random.randn,
            np.random.rand,
            np.random.poisson,
            #  np.random.logistic,
            np.random.exponential,
            #    np.random.binomial
        ]
        i, j = np.random.randint(0, len(distributions), size=2)
        d1 = Distrib(distributions[i])
        d2 = Distrib(distributions[j])
        X = d1(100, 40)
        Y = d2(100, 30)
        # multiply by diagonal matrix with random values
        D = np.diag(np.random.rand(40) * 10)
        X = X @ D
        D = np.diag(np.random.rand(30) * 5)
        Y = Y @ D
        # add a mean vector
        X += np.random.rand(1, 40) * 2
        Y += np.random.rand(1, 30) * 20

        if np.random.rand() < 0.5:
            X, Y = Y, X  # flip dims

        proc = OrthogonalProcrustes()
        score1 = proc.fit(X, Y)
        score2 = proc.score(X, Y)
        assert np.isclose(score1, score2)

        proc = OrthogonalProcrustes(standardize_features=True)
        score1_z_score = proc.fit(X, Y)
        score2_z_score = proc.score(X, Y)
        assert np.isclose(score1_z_score, score2_z_score)
        assert score1 < score1_z_score, (
            f"score1: {score1}, score1_z_score: "
            f"{score1_z_score} - with standardization, expecting a better"
            " score (higher similarity)"
        )

        proc = OrthogonalProcrustes(projection_strategy="pca", n_components=15)
        score1 = proc.fit(X, Y)
        score2 = proc.score(X, Y)
        assert np.isclose(score1, score2)

        proc = OrthogonalProcrustes(projection_strategy="srp", n_components=15)
        score1 = proc.fit(X, Y)
        score2 = proc.score(X, Y)
        assert np.isclose(score1, score2)

        proc = OrthogonalProcrustes(projection_strategy="zero_pad")
        score1 = proc.fit(X, Y)
        score2 = proc.score(X, Y)
        assert np.isclose(score1, score2)

        proc = OrthogonalProcrustes(projection_strategy="incremental_pca")
        score1 = proc.fit(X, Y)
        score2 = proc.score(X, Y)
        assert np.isclose(score1, score2)

        # Test with badly conditioned model features?


def test_invariances():
    X = np.random.randn(100, 20)
    Y = np.random.randn(100, 20)
    proc = OrthogonalProcrustes()

    # Scale invariance
    score1 = proc.fit(X, Y)
    score2 = proc.fit(2 * X, 3 * Y)
    assert np.isclose(score1, score2)

    # Translation invariance
    score3 = proc.fit(X + 1.0, Y + 2.0)
    assert np.isclose(score1, score3)


def test_score_functions():
    X = np.random.randn(100, 20)
    Y = np.random.randn(100, 20)

    # Test Frobenius score bounds
    proc_frob = OrthogonalProcrustes(score_function="frobenius")
    score_frob = proc_frob.fit(X, Y)
    assert -1.0 <= score_frob <= 1.0

    # Test Angular Frobenius score bounds
    proc_ang = OrthogonalProcrustes(score_function="angular_frobenius")
    score_ang = proc_ang.fit(X, Y)
    assert 0 <= score_ang <= np.pi


def test_dimensionality_handling():
    # Test different input dimensions
    X = np.random.randn(100, 30)
    Y = np.random.randn(100, 20)
    X2 = np.random.randn(20, 30)
    Y2 = np.random.randn(20, 20)

    # Test PCA projection
    proc_pca = OrthogonalProcrustes(projection_strategy="pca", n_components=15)
    score_pca = proc_pca.fit(X, Y)
    LOGGER.info(f"score pca: {score_pca}")
    assert proc_pca.R.shape == (15, 15)
    score_pca2 = proc_pca.score(X2, Y2)
    LOGGER.info(f"score pca2: {score_pca2}")

    # Test SRP projection
    proc_srp = OrthogonalProcrustes(projection_strategy="srp", n_components=15)
    score_srp = proc_srp.fit(X, Y)
    LOGGER.info(f"score srp: {score_srp}")
    assert proc_srp.R.shape == (15, 15)
    score_srp2 = proc_srp.score(X2, Y2)
    LOGGER.info(f"score srp2: {score_srp2}")

    # Test zero-padding
    proc_pad = OrthogonalProcrustes(projection_strategy="zero_pad")
    score_pad = proc_pad.fit(X, Y)
    LOGGER.info(f"score pad: {score_pad}")
    assert proc_pad.R.shape == (30, 30)
    score_pad2 = proc_pad.score(X2, Y2)
    LOGGER.info(f"score pad2: {score_pad2}")


def test_dimension_invariant_or_not():
    # IDEA Better data generation would allow to repeat the experiment
    X = np.random.randn(1000, 10)
    Y = np.random.randn(1000, 10)

    scores = []

    proc = OrthogonalProcrustes()
    score = proc.fit(X, Y)
    scores.append(score)
    dims = [10]

    for dim in [15, 20, 40, 100, 200, 400, 1000]:
        upprojection1 = np.random.randn(10, dim)
        upprojection2 = np.random.randn(10, dim)

        X_up = X @ upprojection1
        Y_up = Y @ upprojection2

        proc = OrthogonalProcrustes()

        score_up = proc.fit(X_up, Y_up)
        scores.append(score_up)
        dims.append(dim)

    # plot scores vs dims
    plt.plot(dims, scores)
    os.makedirs("test_plots", exist_ok=True)
    plt.savefig("test_plots/scores_vs_dims_procrustes.png")
    LOGGER.info("saved plot to test_plots/scores_vs_dims_procrustes.png")


def test_orthogonality():
    """R should be orthogonal: R^T R = R R^T = I."""
    X = np.random.randn(100, 20)
    Y = np.random.randn(100, 20)
    proc = OrthogonalProcrustes()
    proc.fit(X, Y)

    Id = np.eye(proc.R.shape[0])
    assert np.allclose(proc.R.T @ proc.R, Id, atol=1e-6)
    assert np.allclose(proc.R @ proc.R.T, Id, atol=1e-6)
    assert np.allclose(np.linalg.det(proc.R) ** 2, 1.0, atol=1e-6)


def test_optimal_rotation():
    """R should minimize ||XR - Y||_F among all orthogonal matrices."""
    X = np.random.randn(100, 20)
    Y = np.random.randn(100, 20)
    proc = OrthogonalProcrustes()
    proc.fit(X, Y)

    for _ in range(10):
        # Generate random orthogonal matrix
        Q = np.linalg.qr(np.random.randn(20, 20))[0]

        # Compare Frobenius norm with optimal R vs random orthogonal Q
        error_optimal = np.linalg.norm(X @ proc.R - Y, "fro")
        error_random = np.linalg.norm(X @ Q - Y, "fro")

        assert error_optimal <= error_random


def test_gradual_similarity_decrease():
    """Test that similarity decreases monotonically as matrices become less similar."""
    X = np.random.randn(100, 20)
    Q = np.linalg.qr(np.random.randn(20, 20))[0]  # Random orthogonal matrix
    Y = X @ Q  # Perfectly similar under orthogonal transform

    proc = OrthogonalProcrustes()
    base_score = proc.fit(X, Y)

    # Gradually add noise to Y
    n_trials = 10
    noise_levels = np.linspace(0, 2, 10)
    scores = np.zeros((len(noise_levels), n_trials))
    scores[0] = base_score
    scores_refit = np.zeros((len(noise_levels), n_trials))
    scores_refit[0] = base_score

    for nidx, noise in enumerate(noise_levels[1:]):
        for i in range(n_trials):
            Y_noisy = Y + noise * np.random.randn(*Y.shape)
            score = proc.score(X, Y_noisy)
            scores[nidx + 1, i] = score
            proc_new = OrthogonalProcrustes()
            scores_refit[nidx + 1, i] = proc_new.fit(X, Y_noisy)

    mean_scores = np.mean(scores, axis=1)
    mean_scores_refit = np.mean(scores_refit, axis=1)
    std_scores = np.std(scores, axis=1)
    std_scores_refit = np.std(scores_refit, axis=1)

    # plot scores vs noise levels
    plt.figure(figsize=(10, 6))
    plt.plot(
        noise_levels, mean_scores, "b-", label="scores from original procrustes fit"
    )
    plt.plot(noise_levels, mean_scores_refit, "r-", label="scores (new fit)")
    plt.fill_between(
        noise_levels,
        mean_scores - std_scores,
        mean_scores + std_scores,
        alpha=0.2,
        color="blue",
    )
    plt.fill_between(
        noise_levels,
        mean_scores_refit - std_scores_refit,
        mean_scores_refit + std_scores_refit,
        alpha=0.2,
        color="red",
    )
    plt.xlabel("Noise Level")
    plt.ylabel("Similarity Score")
    plt.title("Procrustes Similarity vs Noise Level")
    plt.legend()
    plt.grid(True)

    os.makedirs("test_plots", exist_ok=True)
    plt.savefig("test_plots/scores_vs_noise_levels_procrustes.png")

    LOGGER.info("saved plot to test_plots/scores_vs_noise_levels_procrustes.png")

    # Check monotonic decrease
    for i in range(len(mean_scores) - 1):
        assert mean_scores[i] >= mean_scores[i + 1], (
            f"Similarity should decrease "
            f"with noise: {mean_scores[i]} < {mean_scores[i + 1]} at noise level"
            f" {noise_levels[i]}"
        )
        assert mean_scores_refit[i] >= mean_scores_refit[i + 1], (
            f"Similarity should decrease with noise: "
            f"{mean_scores_refit[i]} < {mean_scores_refit[i + 1]} "
            f"at noise level {noise_levels[i]}"
        )


if __name__ == "__main__":
    test_basic_functionality()
    test_score_vs_fit()
    test_invariances()
    test_score_functions()
    test_dimensionality_handling()
    test_dimension_invariant_or_not()
    test_orthogonality()
    test_optimal_rotation()
    test_gradual_similarity_decrease()
