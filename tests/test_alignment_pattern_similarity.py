"""Tests for the alignment-pattern similarity metrics."""

import numpy as np
import pytest

from multitasking.analyses.alignment_patterns.base import (
    SIMILARITY_METRICS,
    get_similarity_function,
    negative_mean_squared_error,
    vectorized_similarity_mean,
)


def test_negative_mean_squared_error_matches_definition():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])
    assert negative_mean_squared_error(x, y) == pytest.approx(-5.0 / 3.0)


def test_mse_is_oriented_so_that_larger_is_more_similar():
    """The whole pipeline takes the argmax of the similarity, so MSE is negated."""
    ref = np.array([0.1, 0.2, 0.3])
    close = ref + 0.01
    far = ref + 0.5
    assert negative_mean_squared_error(ref, close) > negative_mean_squared_error(ref, far)
    assert negative_mean_squared_error(ref, close) <= 0.0


@pytest.mark.parametrize("metric", SIMILARITY_METRICS)
def test_vectorized_similarity_mean_matches_the_pairwise_function(metric):
    rng = np.random.default_rng(0)
    ref = rng.normal(size=8)
    targets = rng.normal(size=(5, 8))
    fn = get_similarity_function(metric)
    if metric == "pearson":
        expected = np.mean([fn(ref, t)[0] for t in targets])
    else:
        expected = np.mean([fn(ref, t) for t in targets])
    assert vectorized_similarity_mean(ref, targets, metric) == pytest.approx(expected)
