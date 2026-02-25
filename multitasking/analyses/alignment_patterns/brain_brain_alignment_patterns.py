"""Brain-brain alignment patterns."""

import warnings
from typing import Callable

import numpy as np
import pandas as pd

from multitasking.analyses.alignment_patterns.base import (
    AlignmentPattern,
    get_similarity_function,
    vectorized_similarity_mean,
)
from multitasking.CONSTANTS import ROIS, SUBJECTS


class PairwiseBrainBrainAlignmentPatterns(AlignmentPattern):
    """Compute pairwise brain–brain alignment patterns.

    Stores alignment patterns between subjects (predictors and targets) across
    ROIs and exposes helpers to derive reference and target patterns as well as
    their similarity.
    """
    def __init__(
        self,
        metric: str,
        main_outcome: str = "score",
    ):
        """Initialize a brain–brain alignment pattern object.

        Args:
            metric: Alignment metric ("rsa" or "linear_predictivity").
            main_outcome: Column to use for alignment patterns, default is "score"
            (i.e., the raw score).
        """
        super().__init__(metric, main_outcome)
        self.alignment_pattern_df: pd.DataFrame = pd.DataFrame()

    def get_alignment_pattern_df(
        self,
        df: pd.DataFrame,
        split: str,
    ) -> None:
        """Extract and store alignment patterns from intersubject dataframe.

        Args:
            df: DataFrame with columns: roi, subject, layer, score, metric, split.
            split: Data split to use (e.g., "train", "test").

        Returns:
            None. The resulting alignment patterns are stored on
            ``self.alignment_pattern_df``.
        """
        if not self.alignment_pattern_df.empty:
            warnings.warn("Alignment pattern data already exists. Not concatenating new data.",
                          stacklevel=2)
        else:
            ap_df = df.query(f"split == '{split}' and metric == '{self.metric}'"
                             ).pivot(index=["model", "subject", "layer"],
                                     columns="roi", values=self.main_outcome)
            self.alignment_pattern_df = ap_df


    def get_alignment_pattern_dict_by_roi(self, rois: list[str]) -> dict[str, np.ndarray]:
        """Organize alignment patterns by (predictor) ROI.

        For each ROI used as a predictor (stored in the ``layer`` index level),
        this extracts the corresponding rows from ``self.alignment_pattern_df``
        and returns their values for the requested ROI columns.

        Args:
            rois: List of ROI names to organize patterns for (used as columns).

        Returns:
            dict[str, np.ndarray]: Maps predictor ROI name to an array whose
            rows correspond to (model, subject, layer) combinations and whose
            columns correspond to the provided ROIs.
        """
        ap_by_roi: dict[str, np.ndarray] = {roi: np.array([]) for roi in rois}

        for roi in rois:
            squeezed = self.alignment_pattern_df.query("layer == @roi")[rois].squeeze()
            ap_by_roi[roi] = np.asarray(squeezed)

        self.alignment_pattern_dict_by_roi = ap_by_roi
        return ap_by_roi

    def get_reference_pattern_dict_by_roi_subject(
        self, rois: list[str], subjects: list[str]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Get reference patterns by ROI and subject.

        Reference patterns are computed from all rows where the given subject
        does not participate as either predictor (model) or target.
        """
        reference_patterns_by_subject: dict[str, dict[str, np.ndarray]] = {
            subject: {
                roi: np.asarray(
                    self.alignment_pattern_df.query(
                        "layer == @roi and model != @subject and subject != @subject"
                    )[rois].squeeze()
                )
                for roi in rois
            }
            for subject in subjects
        }
        return reference_patterns_by_subject

    def get_target_pattern_dict_by_roi_subject(
        self, rois: list[str], subjects: list[str]
        ) -> dict[str, dict[str, np.ndarray]]:
        """Get target patterns by ROI and subject.

        Target patterns are computed from rows where the given subject
        participates as the target subject.
        """
        target_patterns_by_subject: dict[str, dict[str, np.ndarray]] = {
            subject: {
                roi: np.asarray(
                    self.alignment_pattern_df.query(
                        "layer == @roi and subject == @subject"
                    )[rois].squeeze()
                )
                for roi in rois
            }
            for subject in subjects
        }
        return target_patterns_by_subject

    def get_avg_target_pattern_dict_by_roi(
        self, rois: list[str], subjects: list[str]
    ) -> dict[str, np.ndarray]:
        """Get average target patterns by ROI.

        For each ROI, this averages the target patterns across the provided
        subjects, preserving the feature dimension.
        """
        target_patterns_by_roi_subject = self.get_target_pattern_dict_by_roi_subject(rois, subjects)
        avg_target_patterns_by_roi = {
            roi: np.asarray(
                [np.mean(target_patterns_by_roi_subject[subject][roi],
                         axis=0) \
                for subject in subjects])
            for roi in rois
        }
        return avg_target_patterns_by_roi

    def get_brain_brain_alignment_pattern_similarity(
        self,
        rois: list[str] = ROIS,
        subjects: list[str] = SUBJECTS,
        similarity_metric: str = "pearson",
    ) -> pd.DataFrame:
        """Compute similarity between alignment patterns and reference patterns.

        Args:
            rois: List of ROI names to compute similarity for.
            subjects: List of subject names to compute similarity for.
            similarity_metric: Similarity metric. Only ``"pearson"`` is
                supported.

        Returns:
            DataFrame with columns: model, roi, subject, similarity.
        """
        similarities_df = pd.DataFrame(columns=["roi", "subject", "similarity"])
        rows = []
        self.similarity_metric: str = similarity_metric
        self.similarity_function: Callable = get_similarity_function(similarity_metric)
        reference_patterns_by_subject = \
            self.get_reference_pattern_dict_by_roi_subject(
                rois, subjects
                ) # all patterns where subject does not participate (neither predictor nor target)
        target_patterns_by_roi_subject = self.get_target_pattern_dict_by_roi_subject(
            rois, subjects
            ) # all patterns where subject participates as target
        for subject, reference_patterns in reference_patterns_by_subject.items():
            for roi in rois:  # predictor ROI
                avg_reference_pattern = np.mean(reference_patterns[roi], axis=0)
                target_patterns = np.asarray(
                    target_patterns_by_roi_subject[subject][roi]
                )
                similarity = vectorized_similarity_mean(
                    avg_reference_pattern,
                    target_patterns,
                    self.similarity_metric,
                )
                rows.append({"roi": roi, "subject": subject, "similarity": similarity})
        similarities_df = pd.DataFrame(rows)
        self.alignment_pattern_similarity_df = similarities_df
        return similarities_df
