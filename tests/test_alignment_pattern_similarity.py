"""Tests for the alignment-pattern similarity metrics."""

import pickle

import numpy as np
import pytest
from scipy.stats import pearsonr

from multitasking.analyses.alignment_patterns.base import (
    SIMILARITY_METRICS,
    get_similarity_function,
    negative_mean_squared_error,
    vectorized_similarity_mean,
)


def test_negative_mean_squared_error_is_zero_for_identical_patterns():
    pattern = np.array([0.1, 0.5, 0.3])
    assert negative_mean_squared_error(pattern, pattern) == 0.0


def test_negative_mean_squared_error_matches_definition():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])
    assert negative_mean_squared_error(x, y) == pytest.approx(-5.0 / 3.0)


def test_negative_mean_squared_error_is_symmetric():
    x = np.array([0.2, 0.9, -0.4])
    y = np.array([0.1, 0.3, 0.8])
    assert negative_mean_squared_error(x, y) == pytest.approx(
        negative_mean_squared_error(y, x)
    )


def test_mse_is_oriented_so_that_larger_is_more_similar():
    """The whole pipeline takes the argmax of the similarity, so MSE is negated."""
    ref = np.array([0.1, 0.2, 0.3])
    close = ref + 0.01
    far = ref + 0.5
    assert negative_mean_squared_error(ref, close) > negative_mean_squared_error(ref, far)
    assert negative_mean_squared_error(ref, close) <= 0.0


def test_mse_is_sensitive_to_scale_unlike_pearson():
    """A rescaled pattern is identical under pearson but not under MSE."""
    ref = np.array([0.1, 0.2, 0.3])
    scaled = 2 * ref
    assert pearsonr(ref, scaled)[0] == pytest.approx(1.0)
    assert negative_mean_squared_error(ref, scaled) < 0.0


def test_get_similarity_function_dispatches():
    assert get_similarity_function("pearson") is pearsonr
    assert get_similarity_function("mse") is negative_mean_squared_error


def test_get_similarity_function_rejects_unknown_metric():
    with pytest.raises(ValueError, match="Unsupported similarity metric"):
        get_similarity_function("euclidean")


@pytest.mark.parametrize("metric", SIMILARITY_METRICS)
def test_similarity_function_is_picklable(metric):
    """Alignment-pattern objects store the function and are pickled to $SCRATCH."""
    fn = get_similarity_function(metric)
    assert pickle.loads(pickle.dumps(fn)) is fn


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


@pytest.mark.parametrize("metric", SIMILARITY_METRICS)
def test_vectorized_similarity_mean_accepts_a_single_target(metric):
    rng = np.random.default_rng(1)
    ref = rng.normal(size=6)
    target = rng.normal(size=6)
    assert vectorized_similarity_mean(ref, target, metric) == pytest.approx(
        vectorized_similarity_mean(ref, target[None, :], metric)
    )


@pytest.mark.parametrize("metric", SIMILARITY_METRICS)
def test_vectorized_similarity_mean_returns_nan_without_targets(metric):
    ref = np.array([0.1, 0.2, 0.3])
    assert np.isnan(vectorized_similarity_mean(ref, np.array([]), metric))


def test_vectorized_similarity_mean_rejects_unknown_metric():
    ref = np.array([0.1, 0.2, 0.3])
    targets = np.array([[0.1, 0.2, 0.4]])
    with pytest.raises(ValueError, match="Unknown metric"):
        vectorized_similarity_mean(ref, targets, "euclidean")
