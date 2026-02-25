"""Model-brain alignment patterns."""

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from multitasking.analyses.alignment_patterns.base import (
    AlignmentPattern,
    get_similarity_function,
    vectorized_similarity_mean,
)
from multitasking.analyses.alignment_patterns.brain_brain_alignment_patterns import (
    PairwiseBrainBrainAlignmentPatterns,
)


class ModelBrainAlignmentPatterns(AlignmentPattern):
    """Compute alignment patterns between model layers and brain ROIs.

    An alignment pattern for a given model layer (predictor) is a vector of
    scores representing how well that model layer predicts each target ROI.
    The patterns are stored in a dataframe indexed by
    ``(model, subject, layer, predictor)`` with ROIs as columns.
    """

    def __init__(
        self,
        metric: str,
        main_outcome: str = "score",
    ):
        """Initialize a model–brain alignment pattern object.

        Args:
            metric: Alignment metric (currently only "pearson" is supported).
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
            self.split = split
            if "predictor" not in df.columns:
                df["predictor"] = df["model"].astype(str) + "__" + df["layer"].astype(str)
            ap_df = df.query(f"split == '{split}' and metric == '{self.metric}'"
                            ).pivot(index=["model", "subject", "layer", "predictor"],
                                    columns="roi", values=self.main_outcome)
            self.alignment_pattern_df = ap_df

    def get_alignment_pattern_dict_by_predictor(
        self,
        predictors: list[str],
        rois: list[str],
    ) -> dict[str, np.ndarray]:
        """Organize alignment patterns by predictor.

        For each predictor, this method extracts a vector whose
        elements correspond to the scores for the provided ROIs.

        Args:
            predictors: List of predictor identifiers to organize patterns for.
            rois: List of ROI names (columns) to include in the vectors.

        Returns:
            dict[str, np.ndarray]: Maps predictor name to a vector of shape
            ``(n_rois,)``.
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
            similarity_metric: Similarity metric. Only ``"pearson"`` is
                supported.

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
            predictor_patterns_keys = sub_df.index.get_level_values("predictor")
            predictor_patterns_vals = sub_df[rois].values.squeeze()
            predictor_patterns_dict = dict(
                zip(predictor_patterns_keys, predictor_patterns_vals, strict=False)
            )
            for predictor in tqdm(
                predictors, total=len(predictors), desc="predictors", leave=False
            ):
                for roi in rois:
                    predictor_pattern = predictor_patterns_dict[predictor]
                    target_patterns_arr = np.asarray(target_patterns[roi])
                    similarity = vectorized_similarity_mean(
                        predictor_pattern,
                        target_patterns_arr,
                        self.similarity_metric,
                    )
                    rows.append({
                        "roi": roi,
                        "predictor": predictor,
                        "subject": subject,
                        "similarity": similarity,
                    })
        similarities = pd.DataFrame(rows)
        similarities["split"] = self.split
        similarities["metric"] = self.metric
        self.alignment_pattern_similarity_df = similarities

        return similarities
