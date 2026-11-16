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


def test_fold_averaged_r_matches_scipy_and_ignores_affine_target_shift(
    rng: np.random.Generator,
):
    from scipy import stats

    from multitasking.metrics.linear_predictivity import mean_voxel_pearson_r

    x = rng.normal(size=(80, 20))
    w = rng.normal(size=(20, 6))
    y = x @ w + rng.normal(size=(80, 6))
    metric = LinearPredictivity()
    metric(x, y)
    x_test = rng.normal(size=(40, 20))
    y_test = x_test @ w + rng.normal(size=(40, 6))

    r, n_invalid = metric.evaluate_fold_averaged_r(x_test, y_test)

    xn = (x_test - metric.x_mean_train) / metric.x_sd_train
    pred = np.mean([m.predict(xn) for m in metric.models], axis=0)
    yn = (y_test - metric.y_mean_train) / metric.y_sd_train
    expected = np.mean([stats.pearsonr(pred[:, v], yn[:, v])[0] for v in range(6)])
    assert n_invalid == 0
    assert math.isclose(r, expected, abs_tol=1e-6)

    r_affine, _ = metric.evaluate_fold_averaged_r(x_test, 3.0 * y_test + 5.0)
    assert math.isclose(r_affine, r, abs_tol=1e-5)
    r2 = np.mean(metric.evaluate_models(x_test, y_test))
    r2_affine = np.mean(metric.evaluate_models(x_test, 3.0 * y_test + 5.0))
    assert not math.isclose(r2, r2_affine, abs_tol=1e-3)

    # Constant voxels are excluded and counted.
    y_const = y_test.copy()
    y_const[:, 0] = 1.0
    assert mean_voxel_pearson_r(pred, y_const)[1] == 1


def test_linear_predictivity_r_row_leaves_r2_rows_unchanged(
    rng: np.random.Generator, tmp_path
):
    from benedict import benedict

    from multitasking.benchmark import compute_linear_predictivity

    config = benedict({
        "linear_predictivity": {"enabled": True, "seed": 1},
        "dataset": {"name": "t"},
        "feature_extraction": {"model": "m"},
    })
    model_rep = {"train_L": rng.normal(size=(60, 15)),
                 "test_L": rng.normal(size=(30, 15))}
    brain = {"train_R": rng.normal(size=(60, 4)),
             "test_R": rng.normal(size=(30, 4))}
    compute_linear_predictivity(
        model_rep, brain, ["L"], ["R"], ["train", "test"], config,
        output_path=tmp_path, subject_id="s1")

    import pandas as pd
    df = pd.read_csv(tmp_path / "scoresheets" / "scores_linear_predictivity_s1.csv")
    r2 = df[df["metric"] == "linear_predictivity"]
    r = df[df["metric"] == "linear_predictivity_r"]
    assert len(r2) == 2 and len(r) == 2
    assert list(df["metric"][:2]) == ["linear_predictivity"] * 2

    metric = LinearPredictivity(seed=1)
    train_score, _ = metric(model_rep["train_L"], brain["train_R"])
    test_scores = metric.evaluate_models(model_rep["test_L"], brain["test_R"])
    assert math.isclose(r2[r2["split"] == "train"]["score"].item(), train_score,
                        rel_tol=1e-9)
    assert math.isclose(r2[r2["split"] == "test"]["score"].item(),
                        float(np.nanmean(test_scores)), rel_tol=1e-9)
