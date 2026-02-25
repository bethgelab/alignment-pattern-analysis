"""Model-brain alignment patterns."""

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from multitasking.analyses.alignment_patterns.base import (
    AlignmentPattern,
    get_similarity_function,
)
from multitasking.analyses.alignment_patterns.brain_brain_alignment_patterns import (
    PairwiseBrainBrainAlignmentPatterns,
)


class ModelBrainAlignmentPatterns(AlignmentPattern):
    """Model-brain alignment patterns.

    Computes alignment patterns between model layers (predictors) and brain ROIs (targets).
    An alignment pattern for a given model layer is a vector of scores representing how
    well that model layer predicts each target ROI.

    Attributes:
        alignment_pattern_data: AlignmentPatternData
            Structured storage for alignment patterns.
    """

    def __init__(
        self,
        metric: str,
        main_outcome: str = "score",
    ):
        """Initialize ModelBrainAlignmentPatterns.

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
            self.split = split
            if "predictor" not in df.columns:
                df["predictor"] = df["model"].astype(str) + "__" + df["layer"].astype(str)
            ap_df = df.query(f"split == '{split}' and metric == '{self.metric}'"
                            ).pivot(index=["model", "subject", "layer", "predictor"],
                                    columns="roi", values=self.main_outcome)
            self.alignment_pattern_df = ap_df

    def get_alignment_pattern_dict_by_predictor(self, predictors: list[str], rois: list[str]) -> dict[str, np.ndarray]:
        """Organize alignment patterns by ROI.

        Groups alignment patterns by predictor ROI and creates matrices where rows are
        target ROIs and columns are target subjects.

        Args:
            predictors: List of predictor names to organize patterns for.
            rois: List of ROI names to organize patterns for.

        Returns:
            dict[str, np.ndarray]: Maps ROI name to array of shape (n_target_rois, n_subjects_total).
        """
        ap_by_predictor: dict[str, np.ndarray] = {predictor: np.array([]) for predictor in predictors}

        for predictor in predictors:
            matrix = self.alignment_pattern_df.query(
                "predictor == @predictor"
                )[rois].values.squeeze()
            ap_by_predictor[predictor] = matrix

        self.alignment_pattern_dict_by_predictor = ap_by_predictor
        return ap_by_predictor

    def get_alignment_pattern_similarity(
        self,
        subjects: list[str],
        predictors: list[str],
        rois: list[str],
        brain_brain_alignment_patterns: PairwiseBrainBrainAlignmentPatterns,
        similarity_metric: str = "pearson"
    ) -> pd.DataFrame:
        """Compute similarity between model alignment patterns and reference patterns.

        Args:
            subjects: List of subject identifiers.
            predictors: List of predictor identifiers.
            rois: List of ROI names.
            brain_brain_alignment_patterns: PairwiseBrainBrainAlignmentPatterns object.
            similarity_metric: Similarity metric ("pearson" or "mse").

        Returns:
            pd.DataFrame: Structure: roi -> predictor -> subject -> similarity scores
        """
        self.similarity_metric: str = similarity_metric
        self.similarity_function: Callable = get_similarity_function(similarity_metric)
        similarities = pd.DataFrame(columns=["roi", "predictor", "subject", "similarity"])
        rows = []
        target_patterns_by_roi_subject = \
            brain_brain_alignment_patterns.get_target_pattern_dict_by_roi_subject(
                rois, subjects
                )
        for subject, target_patterns in tqdm(
            target_patterns_by_roi_subject.items(),
            total=len(target_patterns_by_roi_subject),
            desc="subjects"):
            sub_df = self.alignment_pattern_df.query("subject == @subject")
            predicor_patterns_keys = sub_df.index.get_level_values("predictor")
            predicor_patterns_vals = sub_df[rois].values.squeeze()
            predicor_patterns_dict = dict(zip(predicor_patterns_keys, predicor_patterns_vals, strict=False))
            for predictor in tqdm(predictors, total=len(predictors), # models
                                  desc="predictors", leave=False):
                for roi in rois:
                    # avg_target_pattern = np.mean(target_patterns[roi], axis=0)

                    rows.append({
                        "roi": roi,
                        "predictor": predictor,
                        "subject": subject,
                        "similarity": np.mean(np.array([self.similarity_function(
                            target_pattern, predicor_patterns_dict[predictor]).statistic for \
                                target_pattern in target_patterns[roi]])
                        )
                    }
                    )
        similarities = pd.DataFrame(rows)
        similarities["split"] = self.split
        similarities["metric"] = self.metric
        self.alignment_pattern_similarity_df = similarities

        return similarities
