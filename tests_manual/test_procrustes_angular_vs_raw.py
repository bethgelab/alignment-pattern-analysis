import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from multitasking.utils.procrustes import OrthogonalProcrustes

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def generate_matrix_pairs(
    n_pairs=1000, n_samples=100, n_features=20, noise_levels=None
):
    """Generate pairs of matrices with varying similarity levels."""
    if noise_levels is None:
        noise_levels = list(np.linspace(0, 2, 20))

    pairs = []
    for _ in range(n_pairs // len(noise_levels)):
        # Create base matrix
        X = np.random.randn(n_samples, n_features)
        X = X / np.linalg.norm(X, "fro")

        for noise in noise_levels:
            # Create Y with controlled dissimilarity
            noise_matrix = np.random.randn(n_samples, n_features)

            # Linear interpolation between X and noise_matrix
            alpha = np.random.rand()  # Normalized mixing coefficient
            Y = (1 - alpha) * X + alpha * noise_matrix
            Y = Y / np.linalg.norm(Y, "fro")

            pairs.append((X, Y))

            # Add some completely random pairs
            if noise > 9:
                Y_random = np.random.randn(n_samples, n_features)
                Y_random = Y_random / np.linalg.norm(Y_random, "fro")
                pairs.append((X, Y_random))

    return pairs


def compute_scores(matrix_pairs):
    frob_scores = []
    angular_scores = []

    for X, Y in matrix_pairs:
        proc_frob = OrthogonalProcrustes(score_function="frobenius")
        proc_ang = OrthogonalProcrustes(score_function="angular_frobenius")

        frob_score = proc_frob.fit(X, Y)
        ang_score = proc_ang.fit(X, Y)

        frob_scores.append(frob_score)
        angular_scores.append(ang_score)

    return np.array(frob_scores), np.array(angular_scores)


def test_procrustes_angular_vs_raw_visual():
    """Doesn't test anything, creates a plot that has to be checked manually."""
    # Generate data and compute scores
    matrix_pairs = generate_matrix_pairs()
    frob_scores, angular_scores = compute_scores(matrix_pairs)

    # Create visualization
    plt.figure(figsize=(12, 8))

    # Main scatter plot
    plt.scatter(frob_scores, angular_scores, alpha=0.5, c="blue")
    plt.xlabel("Frobenius Score (range: [-1, 1])")
    plt.ylabel("Angular Score (range: [0, π])")
    plt.title("Relationship between Frobenius and Angular Procrustes Scores")

    # Add theoretical curve: y = arccos(x)
    x_theory = np.linspace(-1, 1, 100)
    y_theory = np.arccos(x_theory)
    plt.plot(x_theory, y_theory, "r-", label="Theoretical: y = arccos(x)")

    # Add correlation coefficient
    corr, p_value = pearsonr(frob_scores, angular_scores)
    plt.text(
        0.05,
        0.95,
        f"Correlation: {corr:.3f}\np-value: {p_value:.3e}",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("procrustes_angular_vs_raw.png")
    LOGGER.info("saved plot to procrustes_angular_vs_raw.png")


if __name__ == "__main__":
    test_procrustes_angular_vs_raw_visual()
