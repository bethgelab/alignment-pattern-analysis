"""Class for computing orthogonal procrustes alignment score."""

import logging
from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy.linalg import orthogonal_procrustes
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection

LOGGER = logging.getLogger(__name__)


class OrthogonalProcrustes(BaseEstimator):
    """Compute orthogonal procrustes alignment.

    Stores optimal rotation matrix and downprojection matrices internally,
    for fitting once and then re-applying to new data.

    References:
    - [1] Alex H Williams, Erin Kunz, Simon Kornblith, and Scott Linderman. Generalized
      shape metrics on neural representations. Advances in Neural Information Processing
      Systems, 34:4738–4750, 2021.
      https://proceedings.neurips.cc/paper_files/paper/2021/file/252a3dbaeb32e7690242ad3b556e626b-Paper.pdf
      Code: https://github.com/ahwillia/netrep

    - [2] Frances Ding, Jean-Stanislas Denain, and Jacob Steinhardt. Grounding
      representation similarity through statistical testing. Advances in Neural
      Information Processing Systems, 34:1556–1568, 2021.
      https://proceedings.neurips.cc/paper_files/paper/2021/file/0c0bf917c7942b5a08df71f9da626f97-Paper.pdf
      Code: https://github.com/js-d/sim_metric

    - [3] Bo, Yiqing, et al. "Evaluating Representational Similarity Measures from the
      Lens of Functional Correspondence." arXiv preprint arXiv:2411.14633 (2024).
      https://arxiv.org/pdf/2411.14633

    - [4] Cloos, Nathan, et al. "Differentiable optimization of similarity scores
      between models and brains." arXiv preprint arXiv:2407.07059 (2024).
      https://arxiv.org/pdf/2407.07059
      Code: https://github.com/nacloos/diffscore
            Angular Procrustes: https://github.com/nacloos/diffscore/blob/main/diffscore/analysis/measures.py#L319
            - This might be using a Fourier method for downsampling
              of spatial feature maps:
              https://github.com/nacloos/similarity-repository/blob/main/similarity/registry/resi/resi/measures/utils.py#L187
              Called from here: https://github.com/nacloos/similarity-repository/blob/6f30624eb95a57d289f764dbcaa2f9da9b965d08/similarity/registry/resi/resi/measures/procrustes.py#L167
      Newer(?) code: https://github.com/nacloos/similarity-repository
      - https://github.com/nacloos/similarity-repository/blob/6f30624eb95a57d289f764dbcaa2f9da9b965d08/similarity/registry/survey_measures/__init__.py#L25
      - https://github.com/nacloos/similarity-repository/blob/6f30624eb95a57d289f764dbcaa2f9da9b965d08/similarity/registry/resi/resi/measures/procrustes.py#L52
      - also see orthogonal_angular_shape_metric there (angular is for the arccosine)

    """

    def __init__(
        self,
        score_function="frobenius",
        projection_strategy="pca",
        n_components=None,
        projection_seed=None,
        standardize_features=False,
        generalized_procrustes=False,
    ):
        """Initialize the OrthogonalProcrustes class.

        Parameters:

        - score_function: str, what score is applied to the
                            optimally-orthogonally-matched feature matrix pair, to
                            obtain the final score.
                            One of ["frobenius", "angular_frobenius"].
            "frobenius":  trace(A.T @ B) / (||A||_F ||B||_F)
                -- Leads to a score between -1 and 1. (1 is most similar.)
            "angular_frobenius": arccos(  trace(A.T @ B) / (||A||_F ||B||_F)  )
                -- Leads to a _distance_ between 0 and pi. Suggested by [1].
                   (0 is most similar.)

        - projection_strategy: str, how to reduce the dimensionality of the
          feature matrices.
            "pca": perform PCA on the feature matrices.
            "srp": perform Sparse Random Projection on the feature matrices.
            "zero_pad": zero-pad the feature matrices to the same dimensionality.

        - n_components: int, or None. The target number of components in
          dimensionality reduction / up-projection. If None, we use the smaller of
          the two feature spaces for pca and srp, and the larger of the two for
          zero_pad.

        - standardize_features: bool, whether to standardize (z-score) the features.
          This normalizes each feature, while otherwise we just normalize each
           entire matrix by its Frobenius norm.
            - This will be applied BEFORE dimensionality reduction.
              TODO: Discuss whether to flip those two.

        - generalized_procrustes: Superfluous.
            If enabled, allow scaling by a single scalar
            in addition to orthogonal transform.
            Computes a slightly different score, so that the train score of generalized
            procrustes will always be below the one of orthogonal procrustes, so the two
            cannot be compared directly. But its just the square of the score of the
            non-generalized setting.
        """
        assert score_function in ["frobenius", "angular_frobenius"]
        assert projection_strategy in ["pca", "srp", "zero_pad", "incremental_pca"]
        self.score_function = score_function
        self.projection_strategy = projection_strategy
        self.n_components = n_components
        self.projection_seed = projection_seed
        self.standardize_features = standardize_features
        self.generalized_procrustes = generalized_procrustes
        if self.generalized_procrustes:
            if self.projection_strategy != "zero_pad":
                LOGGER.warning(
                    "Generalized Procrustes is supposed to be "
                    "used with zero_pad (see sklearn.spatial.procrustes)"
                )
            if self.standardize_features:
                raise ValueError("Generalized Procrustes replaces standardization")

        self.mean = {"model_features": None, "fmri_data": None}
        self.std = {"model_features": None, "fmri_data": None}

        self.R = None

        self.downproject_model_features = None
        self.downproject_fmri_data = None

    def check_is_fitted(self):
        if self.R is None:
            raise AssertionError(
                "This OrthogonalProcrustes instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        if not self._dimensionality_reduction_was_fit():
            raise AssertionError(
                "Strange behaviour, R was fitted but without fitting "
                "dim reduction first??"
            )

    def _check_consistent_shape_of_projected(self, model_features, fmri_data):
        self.check_is_fitted()
        assert self.R.shape[0] == model_features.shape[1]
        assert self.R.shape[0] == fmri_data.shape[1]
        assert model_features.shape[0] == fmri_data.shape[0]

    def _check_consistent_shape_before_projection(self, model_features, fmri_data):
        assert model_features.shape[0] == fmri_data.shape[0]
        D1 = model_features.shape[1]
        D2 = fmri_data.shape[1]
        if self.projection_strategy in ["incremental_pca"]:
            # I'm not sure which part of the code throws an error, whether
            # incremental or raw pca
            # if self.projection_strategy in ["pca", "incremental_pca"]:
            assert model_features.shape[0] > D1, (
                f"For PCA, we need more samples than "
                f"features. Got {model_features.shape[0]} samples and {D1} features."
            )
            assert fmri_data.shape[0] > D2, (
                f"For PCA, we need more samples than "
                f"features. Got {fmri_data.shape[0]} samples and {D2} features."
            )

        if self.n_components is not None:
            if self.projection_strategy == "zero_pad":
                assert D1 <= self.n_components
                assert D2 <= self.n_components
            else:  # pca or srp down-projects so need larger D1, D2
                assert D1 >= self.n_components
                assert D2 >= self.n_components

    @staticmethod
    def _frob_norm(X):
        return np.linalg.norm(X, "fro")

    class _NoOp:
        # def fit(self, X):
        #     return self
        def transform(self, X):
            return X

    def _fit_dimensionality_reduction(self, model_features, fmri_data):
        """Fit dimensionality reduction."""
        self._check_consistent_shape_before_projection(model_features, fmri_data)

        if self.projection_strategy == "zero_pad":
            return self.dimensionality_reduction(model_features, fmri_data)

        N, D1 = model_features.shape
        D2 = fmri_data.shape[1]
        if self.n_components is not None:
            n_components = self.n_components
            if N < n_components:
                raise ValueError(
                    f"Desired n_components ({n_components}) for projection in "
                    f"OrthogonalProcrustes is larger than the number of "
                    f"samples ({N}) used to fit it."
                )
            LOGGER.info(
                f"Projecting down to {n_components} components before procrustes."
            )
        else:
            n_components = np.min([D1, D2])
        if N < n_components:
            self.n_components = N
            n_components = N
            LOGGER.warning(
                f"Reducing the number of components for dimensionality reduction"
                f"from {np.min([D1, D2])} to {N} because we only have {N} samples."
            )

        cls = (
            PCA
            if self.projection_strategy in ["pca", "incremental_pca"]
            else SparseRandomProjection
        )

        self.downproject_model_features = cls(
            n_components=n_components, random_state=self.projection_seed
        )
        self.downproject_fmri_data = cls(
            n_components=n_components, random_state=self.projection_seed
        )

        if n_components < D1:
            self.downproject_model_features.fit(model_features)
        else:
            self.downproject_model_features = self._NoOp()

        if n_components < D2:
            self.downproject_fmri_data.fit(fmri_data)
        else:
            self.downproject_fmri_data = self._NoOp()

        return (
            self.downproject_model_features.transform(model_features),
            self.downproject_fmri_data.transform(fmri_data),
        )

    def _dimensionality_reduction_was_fit(self):
        if self.projection_strategy == "zero_pad":
            return True
        else:
            return (self.downproject_model_features is not None) and (
                self.downproject_fmri_data is not None
            )

    def dimensionality_reduction(
        self, model_features, fmri_data
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Apply (previously fitted) dimensionality reduction.

        Zero padding taken from here:
        https://github.com/nacloos/similarity-repository/blob/main/similarity/registry/resi/resi/measures/utils.py#L125

        Parameters:
        - model_features: np.ndarray, shape (n_samples, n_features)
        - fmri_data: np.ndarray, shape (n_samples, n_features)

        Returns:
        - model_features: np.ndarray, shape (n_samples, n_features)
        - fmri_data: np.ndarray, shape (n_samples, n_features)

        """
        if not self._dimensionality_reduction_was_fit():
            raise AssertionError(
                "Dimensionality reduction was not fit, but was needed "
                "for this operation."
            )
            # self._fit_dimensionality_reduction(model_features, fmri_data)

        self._check_consistent_shape_before_projection(model_features, fmri_data)

        D1 = model_features.shape[1]
        D2 = fmri_data.shape[1]

        if (self.projection_strategy == "zero_pad") and (self.n_components is not None):
            raise ValueError("max_components is not supported for zero_pad strategy")

        if self.projection_strategy == "zero_pad":
            if D1 - D2 == 0:
                return model_features, fmri_data
            elif D1 - D2 > 0:
                return model_features, np.concatenate(
                    (fmri_data, np.zeros((fmri_data.shape[0], D1 - D2))), axis=1
                )
            else:
                return np.concatenate(
                    (model_features, np.zeros((model_features.shape[0], D2 - D1))),
                    axis=1,
                ), fmri_data

        elif self.projection_strategy in ["pca", "incremental_pca", "srp"]:
            return self.downproject_model_features.transform(
                model_features
            ), self.downproject_fmri_data.transform(fmri_data)

        else:
            raise NotImplementedError()

    def fit(self, model_features, fmri_data):
        """Compute the orthogonal procrustes alignment.

        The fitted R will transform the model features to the fmri data.

        Steps:
        1. Center each feature
        1b. Optionally standardize each feature
        2. Project either model features or fmri data to the dimensionality of the other
          feature space. To ensure comparability, set a desired dim in __init__.
        3. Normalize the model features and the fmri data.

        Notes:
          - [1] centers and just normalizes the matrix by its Frobenius norm.
          - [4], older code, whitens instead, while in later versions of the code, [4]
            drops whitening (also does not normalize), but they have an additional
            metric which does normalize (by Frob norm as in [1]). So use that.
        4. Compute optimal rotation matrix using orthogonal_procrustes.
        5. Compute a score based on that.
        6. Return the score.


        Parameters:
        - model_features: np.ndarray, shape (n_samples, n_features)
        - fmri_data: np.ndarray, shape (n_samples, n_features)


        Returns:
        - score: float, orthogonal procrustes alignment score.

        Todo: Check whether sklearn's fit returns self or the score or sth else.
        """
        # -- Center each feature --
        self.mean["model_features"] = np.nanmean(model_features, axis=0, keepdims=True)
        self.mean["fmri_data"] = np.nanmean(fmri_data, axis=0, keepdims=True)
        model_features = model_features - self.mean["model_features"]
        fmri_data = fmri_data - self.mean["fmri_data"]

        # -- Optionally, standardize each feature --
        # --- Edit: This might make Procrustes equivalent to
        #         linear regression (without regularization),
        #         at least according to:
        #         Barbosa, Joao, et al. "Quantifying Differences in Neural Population
        #           Activity With Shape Metrics." bioRxiv (2025): 2025-01.
        if self.standardize_features:
            raise AssertionError(
                "Ensure this is switched off for now, as "
                "we already normalize the data across features globally / "
                "the data is stored normalized on disk (todo drop the "
                "'standardize_features' parameter at some point?)"
            )

        # -- Dimensionality reduction --
        model_features, fmri_data = self._fit_dimensionality_reduction(
            model_features, fmri_data
        )

        # -- Divide by Frobenius norm (necessary to get comparable scores) --

        model_features = model_features / self._frob_norm(model_features)
        fmri_data = fmri_data / self._frob_norm(fmri_data)
        ## self._frob_norm() == np.linalg.norm(), Frob is the default.

        # -- Compute optimal rotation matrix --
        self.R, score = orthogonal_procrustes(model_features, fmri_data)
        # score:  sum of singular values of (A^TB)

        # # double check the score
        # score3 = np.trace(self.R.T @ model_features.T @ fmri_data)
        # # score3:  trace((AR)^TB) = sum of singular values of (A^TB)
        # assert np.isclose(score, score3), (
        #     f"score: {score}, score3: {score3}  - should be close"
        # )

        if self.generalized_procrustes:
            # -- Copied from the end of scipy.spatial.procrustes --
            # Taking into account the scale: score = tr(A.T @ B @ R)
            #   = sum of singular values of (A^TB).
            #  Optimizing for the best scale to align A and B (in terms of MSE),
            #  the score would be that / tr(B.T @ B) but B is already normalized
            #  here, so we can just use the score to align the two normalized
            #  matrices.
            self.scale = score
            fmri_data = np.dot(fmri_data, self.R.T) * score
            disparity = np.sum(np.square(model_features - fmri_data))
            # From gemini-2.5-pro:
            # " The final disparity score returned by this function is
            #   mathematically equivalent to 1 - s^2."
            # (it did explain why, you can derive it)
            assert np.isclose(disparity, 1 - self.scale**2), (
                f"disparity: {disparity}, scale: {self.scale} - should be close"
            )
            score = 1 - disparity

        # -- Compute score --
        # IDEA does it make sense to allow that for generalized
        # procrustes? the score is also in [0,1] so it works, but...
        if self.score_function == "angular_frobenius":
            score = np.arccos(score)
        elif self.score_function == "frobenius":
            pass
        elif self.score_function != "frobenius":
            raise NotImplementedError(f"Invalid score function: {self.score_function}")

        return score

    def transform(self, model_features, fmri_data):
        """Transform the model features and fmri data.

        Steps:
        - center both
        - downproject both
        - normalize both
        - transform model features with optimal R
        """
        self.check_is_fitted()
        self._check_consistent_shape_before_projection(model_features, fmri_data)

        # -- Centering --
        model_features = model_features - self.mean["model_features"]
        fmri_data = fmri_data - self.mean["fmri_data"]

        # -- Optionally, standardize each feature --
        if self.standardize_features:
            model_features = model_features / self.std["model_features"]
            fmri_data = fmri_data / self.std["fmri_data"]

        # -- Dimensionality reduction --
        model_features, fmri_data = self.dimensionality_reduction(
            model_features, fmri_data
        )

        self._check_consistent_shape_of_projected(model_features, fmri_data)

        # -- Normalization --
        model_features = model_features / self._frob_norm(model_features)
        fmri_data = fmri_data / self._frob_norm(fmri_data)

        if self.generalized_procrustes:
            fmri_data = fmri_data * self.scale

        # -- Transform model features with optimal R --
        return model_features @ self.R, fmri_data

    def score(self, model_features, fmri_data):
        features_transformed, fmri_data_transformed = self.transform(
            model_features, fmri_data
        )

        if self.generalized_procrustes:
            disparity = np.sum(np.square(features_transformed - fmri_data_transformed))
            score = 1 - disparity  # is it == 1 - s^2 as well for test data?
        else:
            if "frobenius" in self.score_function:
                score = np.trace(features_transformed.T @ fmri_data_transformed)
            else:
                raise ValueError(f"Invalid score function: {self.score_function}")

        if self.score_function == "angular_frobenius":
            score = np.arccos(score)
        return score

    def fit_and_score(self, model_features, fmri_data):
        self.fit(model_features, fmri_data)
        return self.score(model_features, fmri_data)
