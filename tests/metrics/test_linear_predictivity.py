"""Unit tests for the linear predictivity metric."""

import math

import numpy as np
import pytest

from multitasking.metrics import LinearPredictivity


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1)


@pytest.fixture
def wide_alpha_grid(monkeypatch: pytest.MonkeyPatch):
    """Use a wider alpha grid so CV can choose near-zero regularization."""
    import sklearn.linear_model as sklinear

    def ridgecv_factory(*args, **kwargs):
        kwargs = {**kwargs, "alphas": np.logspace(-6, 6, 25)}
        return sklinear.RidgeCV(**kwargs)

    monkeypatch.setattr(
        "multitasking.metrics.linear_predictivity.RidgeCV", ridgecv_factory
    )


def test_linear_predictivity_returns_float_and_dict(rng: np.random.Generator):
    features1 = rng.normal(size=(10, 100))
    features2 = rng.normal(size=(10, 100))

    linear_predictivity = LinearPredictivity()
    score, details = linear_predictivity(features1, features2)

    assert isinstance(score, float)
    assert isinstance(details, dict)


def test_linear_predictivity_returns_expected_details(rng: np.random.Generator):
    # This tests as a mark to remind you to write tests for the details dictionary
    # once you add more details to the metric. Don't you dare to just delete this test!
    features1 = rng.normal(size=(10, 100))
    features2 = rng.normal(size=(10, 100))

    linear_predictivity = LinearPredictivity()
    _, details = linear_predictivity(features1, features2)

    # Expect details to include trained models and their selected alphas
    assert isinstance(details, dict)
    assert "models" in details
    assert "alphas" in details
    assert isinstance(details["models"], list)
    assert isinstance(details["alphas"], list)
    assert len(details["models"]) == linear_predictivity.num_folds
    assert len(details["alphas"]) == linear_predictivity.num_folds
    assert all(np.isscalar(a) for a in details["alphas"])


def test_linear_predictivity_for_identical_features_returns_1(
    rng: np.random.Generator, wide_alpha_grid
):
    # We need a large number of samples to ensure that the linear model does not
    # overfit. For the case of overfitting, the score would be smaller than 1.0.
    features1 = rng.normal(size=(256, 100))
    features2 = features1.copy()

    linear_predictivity = LinearPredictivity()
    score, _ = linear_predictivity(features1, features2)
    assert math.isclose(score, 1.0, abs_tol=1e-6)


def test_linear_predictivity_is_invariant_to_channel_permutation(
    rng: np.random.Generator, wide_alpha_grid
):
    features1 = rng.normal(size=(256, 100))
    features2 = rng.permutation(features1, axis=1)

    linear_predictivity = LinearPredictivity()
    score, _ = linear_predictivity(features1, features2)
    assert math.isclose(score, 1.0, abs_tol=1e-6)


def test_linear_predictivity_is_invariant_to_channel_scaling(
    rng: np.random.Generator, wide_alpha_grid
):
    features1 = rng.normal(size=(256, 100))
    features2 = features1 * 2

    linear_predictivity = LinearPredictivity()
    score, _ = linear_predictivity(features1, features2)

    assert math.isclose(score, 1.0, abs_tol=1e-6)


def test_linear_predictivity_is_invariant_to_additional_channels(
    rng: np.random.Generator, wide_alpha_grid
):
    features2 = rng.normal(size=(256, 100))
    additional_channels = rng.normal(size=(256, 10))
    features1 = np.concatenate([features2, additional_channels], axis=1)

    linear_predictivity = LinearPredictivity()
    score, _ = linear_predictivity(features1, features2)

    assert math.isclose(score, 1.0, abs_tol=1e-6)


def test_linear_predictivity_returns_nan_for_nan_features1(rng: np.random.Generator):
    features1 = rng.normal(size=(256, 100))
    features2 = rng.normal(size=(256, 100))
    features1[4, 4] = np.nan

    linear_predictivity = LinearPredictivity()
    score, _ = linear_predictivity(features1, features2)

    assert math.isnan(score)


def test_linear_predictivity_returns_nan_for_nan_features2(rng: np.random.Generator):
    features1 = rng.normal(size=(256, 100))
    features2 = rng.normal(size=(256, 100))
    features2[4, 4] = np.nan

    linear_predictivity = LinearPredictivity()
    score, _ = linear_predictivity(features1, features2)

    assert math.isnan(score)


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
def test_linear_predictivity_returns_nan_when_normalizing_constant_features(
    rng: np.random.Generator,
):
    features1 = np.ones((256, 100))
    features2 = rng.normal(size=(256, 100))

    linear_predictivity = LinearPredictivity(normalize=True)
    score, _ = linear_predictivity(features1, features2)

    assert math.isnan(score)

    linear_predictivity = LinearPredictivity(normalize=False)
    score, _ = linear_predictivity(features1, features2)

    assert not math.isnan(score)


def test_evaluate_models_raises_without_fit(rng: np.random.Generator):
    linear_predictivity = LinearPredictivity()
    with pytest.raises(RuntimeError):
        _ = linear_predictivity.evaluate_models(
            rng.normal(size=(10, 3)), rng.normal(size=(10, 2))
        )


def test_evaluate_models_identity_on_holdout_returns_ones(
    rng: np.random.Generator, wide_alpha_grid
):
    # Train on identity mapping
    train_x = rng.normal(size=(512, 100))
    train_y = train_x.copy()
    metric = LinearPredictivity()
    _score, _details = metric(train_x, train_y)

    # Evaluate on a disjoint holdout set with the same mapping
    test_x = rng.normal(size=(256, 100))
    test_y = test_x.copy()
    scores = metric.evaluate_models(test_x, test_y)

    assert isinstance(scores, list)
    assert len(scores) == metric.num_folds
    for s in scores:
        assert math.isclose(float(s), 1.0, abs_tol=1e-6)


def test_evaluate_models_output_types_and_length(rng: np.random.Generator):
    # Basic smoke test on random data to ensure correct structure
    x = rng.normal(size=(64, 20))
    y = rng.normal(size=(64, 5))
    metric = LinearPredictivity(num_folds=4)
    _score, _details = metric(x, y)
    scores = metric.evaluate_models(x, y)

    assert isinstance(scores, list)
    assert len(scores) == 4
    assert all(isinstance(v, float) for v in scores)
