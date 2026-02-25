"""Unit tests for the RSA metric."""

import math

import numpy as np
import pytest
import torch

from multitasking.metrics import RSA
from multitasking.metrics.rsa import _compute_rdm_scipy, _compute_rdm_torch


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1)


def test_compute_rdm_torch_matches_scipy_implementation(rng: np.random.Generator):
    x = rng.normal(size=(1000, 100))
    x_torch = torch.from_numpy(x).to(device="cuda")

    rdm_pytorch = _compute_rdm_torch(x_torch, chunk_size=100).cpu().numpy()
    rdm_scipy = _compute_rdm_scipy(x)

    assert np.allclose(rdm_pytorch, rdm_scipy, atol=1e-6)


def test_rsa_returns_float_and_dict(rng: np.random.Generator):
    features1 = rng.normal(size=(10, 100))
    features2 = rng.normal(size=(10, 100))

    rsa = RSA()
    score, details = rsa(features1, features2)

    assert isinstance(score, float)
    assert isinstance(details, dict)


def test_rsa_returns_rdms_in_details(rng: np.random.Generator):
    features1 = rng.normal(size=(10, 100))
    features2 = rng.normal(size=(10, 100))

    rsa = RSA()
    _, details = rsa(features1, features2)

    assert "rdm1" in details
    assert "rdm2" in details

    assert details["rdm1"].shape == (10, 10)
    assert details["rdm2"].shape == (10, 10)


def test_rsa_for_identical_features_returns_1(rng: np.random.Generator):
    features1 = rng.normal(size=(10, 100))
    features2 = features1.copy()

    rsa = RSA()
    score, _ = rsa(features1, features2)

    assert math.isclose(score, 1.0, abs_tol=1e-6)


def test_rsa_is_invariant_to_channel_permutation(rng: np.random.Generator):
    features1 = rng.normal(size=(10, 100))
    features2 = rng.permutation(features1, axis=1)

    rsa = RSA()
    score, _ = rsa(features1, features2)

    assert math.isclose(score, 1.0, abs_tol=1e-6)


def test_rsa_is_invariant_to_channel_scaling(rng: np.random.Generator):
    features1 = rng.normal(size=(10, 100))
    features2 = features1 * 2

    rsa = RSA()
    score, _ = rsa(features1, features2)

    assert math.isclose(score, 1.0, abs_tol=1e-6)


def test_rsa_is_symmetric(rng: np.random.Generator):
    features1 = rng.normal(size=(10, 100))
    features2 = rng.normal(size=(10, 100))

    rsa = RSA()
    score1, _ = rsa(features1, features2)
    score2, _ = rsa(features2, features1)

    assert math.isclose(score1, score2)


def test_rsa_score_for_simple_example(rng: np.random.Generator):
    features1 = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
    ])

    # rdm1 = [
    #     [0, 0, 1],
    #     [0, 0, 1],
    #     [1, 1, 0],
    # ]

    features2 = np.array([
        [1, 0],
        [0, 1],
        [0, 1],
    ])

    # rdm2 = [
    #     [0, 1, 1],
    #     [1, 0, 0],
    #     [1, 0, 0],
    # ]

    rsa = RSA()
    score, _ = rsa(features1, features2)

    # From calculating Pearson correlation between upper triangular values of rdm1
    # (0, 0, 1) and upper triangular values of rdm2 (1, 1, 0) by hand, we get:
    # score = (-1/3) / (2/3) = -0.5
    assert math.isclose(score, -0.5, abs_tol=1e-6)


def test_rsa_is_nan_for_nan_features(rng: np.random.Generator):
    features1 = rng.normal(size=(10, 100))
    features2 = rng.normal(size=(10, 100))
    features2[4, 4] = np.nan

    rsa = RSA()
    score, _ = rsa(features1, features2)

    assert math.isnan(score)
