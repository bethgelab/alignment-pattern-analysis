"""Linear predictivity metric."""

import re
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from multitasking.metrics.base import Metric


class LinearPredictivity(Metric):
    """Linear predictivity metric.

    This will split the data into folds and fit a linear model from features1 to
    features2 on the training data for each fold. The score is the average of the
    R^2 scores for the respective test data.

    The parameter alpha is fitted on the training data for each fold, using efficient
    leave-one-out cross-validation (see `sklearn.linear_model.RidgeCV`). This means,
    that a different value could be used for each fold. This is in line with the method
    described in [Tang et al., 2024](https://www.biorxiv.org/content/10.1101/2025.07.22.664908v1).
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
                if re.match(r"Input (X|y) contains NaN.", str(error)):
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
            "alphas": [m.alpha_.item() for m in models],
            "train_scores": train_scores,
            "ridgecv_val_mse": ridgecv_val_mse,
            "train_mse": train_mse,
            "test_mse": test_mse,
        }

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
                if re.match(r"Input contains NaN.", str(error)):
                    score = np.nan
                else:
                    raise error
            scores.append(score)
        return scores
