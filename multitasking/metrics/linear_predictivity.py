"""Linear predictivity metric."""

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from multitasking.metrics.base import Metric


def mean_voxel_pearson_r(
    prediction: npt.ArrayLike, target: npt.ArrayLike
) -> tuple[float, int]:
    """Mean over voxels (columns) of the Pearson r between prediction and target.

    Voxels with zero variance in the prediction or the target have an undefined
    r; they are excluded from the mean (nanmean) and counted.

    Parameters:
        prediction: Array of shape (n_samples, n_voxels).
        target: Array of shape (n_samples, n_voxels).

    Returns:
        The mean r (NaN if no voxel has a defined r) and the number of voxels
        with an undefined r.
    """
    p = np.asarray(prediction, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    p = p - p.mean(axis=0)
    t = t - t.mean(axis=0)
    pp = (p * p).sum(axis=0)
    tt = (t * t).sum(axis=0)
    valid = (pp > 0) & (tt > 0)
    r = np.full(p.shape[1], np.nan)
    r[valid] = (p * t).sum(axis=0)[valid] / np.sqrt(pp[valid] * tt[valid])
    n_invalid = int((~valid).sum())
    if n_invalid == len(r):
        return float("nan"), n_invalid
    return float(np.nanmean(r)), n_invalid


class LinearPredictivity(Metric):
    """Linear predictivity metric.

    This will split the data into folds and fit a linear model from features1 to
    features2 on the training data for each fold. The score is the average of the
    R^2 scores for the respective test data.

    The parameter alpha is fitted on the training data for each fold, using efficient
    leave-one-out cross-validation (see `sklearn.linear_model.RidgeCV`). This means,
    that a different value could be used for each fold. This is in line with the method
    described in [Tang et al., 2025](https://www.biorxiv.org/content/10.1101/2025.07.22.664908v1).

    A voxel-mean Pearson r, which ignores offset and scale mismatches, is computed
    exactly like R^2: per fold model, then averaged over folds.
    """

    def __init__(self, normalize: bool = True, num_folds: int = 5, seed: int = 1):
        """Initialize the linear predictivity metric.

        Parameters:
            normalize: Whether to normalize the features before fitting the linear
                model.
            num_folds: The number of folds to use for cross-validation.
            seed: The seed for the random number generator.
        """
        self.normalize = normalize
        self.num_folds = num_folds
        self.seed = seed
        self.models: list[RidgeCV] = []

    def __call__(
        self, features1: npt.ArrayLike, features2: npt.ArrayLike
    ) -> tuple[float, dict[str, Any]]:
        """Compute the linear predictivity score.

        Parameters:
            features1: The first feature matrix of shape (n_samples, n_features_1).
            features2: The second feature matrix of shape (n_samples, n_features_2).

        Returns:
            score: The linear predictivity score.
            details: A dictionary containing:
                - the trained models,
                - their alphas,
                - train scores: scores on the train split for each fold
                - the ridgecv validation MSE (mean over RidgeCV internal
                  leave-one-out single samples)
                - "train_mse": MSE on the train split for each fold
                - "test_mse": MSE on the test split for each fold.
        """
        x = np.asarray(features1, dtype=np.float32)
        y = np.asarray(features2, dtype=np.float32)
        models = []

        if self.normalize:
            self.x_mean_train = x.mean(axis=0)
            self.x_sd_train = x.std(axis=0)
            self.y_mean_train = y.mean(axis=0)
            self.y_sd_train = y.std(axis=0)
            x = (x - x.mean(axis=0)) / x.std(axis=0)
            y = (y - y.mean(axis=0)) / y.std(axis=0)

        kfold = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)

        scores = []
        train_scores = []
        r_scores = []
        r_n_invalid = 0

        ridgecv_val_mse = []
        train_mse = []
        test_mse = []

        for train_index, test_index in kfold.split(x):
            x_train = x[train_index]
            y_train = y[train_index]
            x_test = x[test_index]
            y_test = y[test_index]

            model = RidgeCV(alphas=np.logspace(0, 9, 19),
                            store_cv_results=True,
                            # ensures cv_results_ contains MSE per point
                            )

            try:
                model.fit(x_train, y_train)
            except ValueError as error:
                if "contains NaN" in str(error):
                    return np.nan, {"models": None,
                                    "alphas": [],
                                    "train_scores": [],
                                    "ridgecv_val_mse": [],
                                    "train_mse": [],
                                    "test_mse": [],}
                elif x_train.shape[-1] == 0 or y_train.shape[-1] == 0:
                    return np.nan, {"models": None,
                                    "alphas": [],
                                    "train_scores": [],
                                    "ridgecv_val_mse": [],
                                    "train_mse": [],
                                    "test_mse": [],}
                raise error

            score = model.score(x_test, y_test)
            r, n_invalid = mean_voxel_pearson_r(model.predict(x_test), y_test)
            r_scores.append(r * len(test_index))
            r_n_invalid = max(r_n_invalid, n_invalid)
            score_train = model.score(x_train, y_train)

            scores.append(score * len(test_index))
            train_scores.append(score_train)
            models.append(model)

            # also compute MSE values, for debugging,
            # to have a chance to detect overfitting
            ridgecv_val_mse_ = np.mean(model.cv_results_)
            train_mse_ = np.mean((y_train - model.predict(x_train))**2)
            test_mse_ = np.mean((y_test - model.predict(x_test))**2)
            ridgecv_val_mse.append(ridgecv_val_mse_.item())
            train_mse.append(train_mse_.item())
            test_mse.append(test_mse_.item())

        self.models = models

        return np.sum(scores) / len(x), {
            "models": models,
            "alphas": [float(m.alpha_) for m in models],
            "train_scores": train_scores,
            "ridgecv_val_mse": ridgecv_val_mse,
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_r": np.sum(r_scores) / len(x),
            "train_r_n_invalid": r_n_invalid,
        }

    def evaluate_models_r(
        self, test_features1: npt.ArrayLike, test_features2: npt.ArrayLike
    ) -> tuple[list[float], int]:
        """Voxel-mean Pearson r of each trained model, like `evaluate_models`.

        Parameters:
            test_features1: Feature matrix of shape (n_samples, n_features_1).
            test_features2: Target matrix of shape (n_samples, n_voxels).

        Returns:
            One r per model (NaN on NaN input), and the largest number of voxels
            with undefined r in any model (see `mean_voxel_pearson_r`).

        Raises:
            RuntimeError: If no trained models are found.
        """
        if not self.models:
            raise RuntimeError(
                "No trained models found. Please fit the model before evaluation."
            )
        x = np.asarray(test_features1, dtype=np.float32)
        y = np.asarray(test_features2, dtype=np.float32)
        if self.normalize:
            x = (x - self.x_mean_train) / self.x_sd_train
            y = (y - self.y_mean_train) / self.y_sd_train
        scores = []
        n_invalid = 0
        for m in self.models:
            try:
                r, n = mean_voxel_pearson_r(m.predict(x), y)
            except ValueError as error:
                if "contains NaN" in str(error):
                    r, n = float("nan"), 0
                else:
                    raise
            scores.append(r)
            n_invalid = max(n_invalid, n)
        return scores, n_invalid

    def evaluate_models(
        self, test_features1: npt.ArrayLike, test_features2: npt.ArrayLike
    ) -> list[float]:
        """Evaluate trained models on the a different set of features.

        Parameters:
            test_features1: The first feature matrix of shape
                (n_samples, n_features_1).
            test_features2: The second feature matrix of shape
                (n_samples, n_features_2).

        Returns:
            score: a list of scores (1 for each model)

        Raises:
            RuntimeError: If no trained models are found.
        """
        if not self.models:
            raise RuntimeError(
                "No trained models found. Please fit the model before evaluation."
            )

        x = np.asarray(test_features1, dtype=np.float32)
        y = np.asarray(test_features2, dtype=np.float32)

        if self.normalize:
            x = (x - self.x_mean_train) / self.x_sd_train
            y = (y - self.y_mean_train) / self.y_sd_train

        scores = []
        for m in self.models:
            try:
                score = m.score(x, y)
            except ValueError as error:
                if "contains NaN" in str(error):
                    score = np.nan
                else:
                    raise error
            scores.append(score)
        return scores
