"""Brain-brain alignment patterns."""

import warnings
from typing import Callable

import numpy as np
import pandas as pd

from multitasking.analyses.alignment_patterns.base import (
    AlignmentPattern,
    get_similarity_function,
)
from multitasking.CONSTANTS import ROIS, SUBJECTS


class PairwiseBrainBrainAlignmentPatterns(AlignmentPattern):
    def __init__(
        self,
        metric: str,
        main_outcome: str = "score",
    ):
        """Initialize BrainBrainAlignmentPatterns.

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
        """Extract alignment patterns from intersubject dataframe.

        Args:
            df: DataFrame with columns: roi, subject, layer, score, metric, split.
            split: Data split to use (e.g., "train", "test").
            return_alignment_pattern: If True, return the data; if False, store and return None.

        Returns:
            AlignmentPatternData (or AlignmentPatternDict for backward compat) if return_alignment_pattern is True, else None.
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
        """Organize alignment patterns by ROI.

        Groups alignment patterns by predictor ROI and creates matrices where rows are
        target ROIs and columns are target subjects.

        Args:
            rois: List of ROI names to organize patterns for.

        Returns:
            dict[str, np.ndarray]: Maps ROI name to array of shape (n_target_rois, n_subjects_total).
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
        """Get reference patterns by ROI and subject."""
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
        """Get target patterns by ROI and subject."""
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
        """Get average target patterns by ROI."""
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
            similarity_metric: Similarity metric ("pearson" or "mse").
            all_layers: Whether to compute similarity for all layers (ROIs).

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
            for roi in rois: # predictor ROI
                avg_reference_pattern = np.mean(reference_patterns[roi], axis=0)
                rows.append(
                    {
                    "roi": roi,
                    "subject": subject,
                    "similarity": np.mean(
                        np.array([self.similarity_function(
                            avg_reference_pattern,
                            target_pattern).statistic for target_pattern \
                                in target_patterns_by_roi_subject[subject][roi]])
                    )
                }
                )
        similarities_df = pd.DataFrame(rows)
        self.alignment_pattern_similarity_df = similarities_df
        return similarities_df
