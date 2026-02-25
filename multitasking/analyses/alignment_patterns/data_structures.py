"""Data structures for alignment patterns.

This module provides cleaner alternatives to nested dictionaries for storing
and accessing alignment pattern data.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class AlignmentPatternData:
    """Structured storage for alignment patterns using a DataFrame.

    This replaces the nested dictionary structure:
        dict[str, dict[str, dict[str, float]]]
    with a more maintainable DataFrame-based approach.

    Attributes:
        df: DataFrame with columns: predictor_subject, predictor_roi, target_subject, target_roi, score
    """

    df: pd.DataFrame

    def __init__(self, df: Optional[pd.DataFrame] = None):
        """Initialize AlignmentPatternData.

        Args:
            df: DataFrame with columns (predictor_subject, predictor_roi, target_subject, target_roi, score).
                If None, creates an empty DataFrame with the correct structure.
        """
        if df is None:
            self.df = pd.DataFrame(columns=["predictor_subject", "predictor_roi", "target_subject", "target_roi", "score"])
        else:
            required_cols = {"predictor_subject", "predictor_roi", "target_subject", "target_roi", "score"}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"DataFrame must have columns: {required_cols}")
            self.df = df.copy()

    @classmethod
    def from_nested_dict(
        cls, nested_dict: dict[str, dict[str, dict[str, float]]]
    ) -> "AlignmentPatternData":
        """Create AlignmentPatternData from nested dictionary.

        Args:
            nested_dict: Structure: predictor (subject__roi) -> target_subject -> target_roi -> score

        Returns:
            AlignmentPatternData instance.
        """
        rows = []
        for predictor, subject_dict in nested_dict.items():
            # Parse predictor as "subject__roi" format
            if "__" in predictor:
                predictor_subject, predictor_roi = predictor.split("__", 1)
            else:
                # For backward compatibility, assume predictor is just the ROI
                predictor_subject = ""
                predictor_roi = predictor

            for target_subject, target_dict in subject_dict.items():
                for target_roi, score in target_dict.items():
                    rows.append({
                        "predictor_subject": predictor_subject,
                        "predictor_roi": predictor_roi,
                        "target_subject": target_subject,
                        "target_roi": target_roi,
                        "score": score,
                    })
        return cls(pd.DataFrame(rows))

    def to_nested_dict(self) -> dict[str, dict[str, dict[str, float]]]:
        """Convert to nested dictionary format (for backward compatibility).

        Returns:
            Nested dictionary: predictor (subject__roi) -> target_subject -> target_roi -> score
        """
        result: dict[str, dict[str, dict[str, float]]] = {}
        for _, row in self.df.iterrows():
            predictor_subject = str(row["predictor_subject"])
            predictor_roi = str(row["predictor_roi"])
            predictor = f"{predictor_subject}__{predictor_roi}" if predictor_subject else predictor_roi
            target_subject = str(row["target_subject"])
            target_roi = str(row["target_roi"])
            score = float(row["score"])

            if predictor not in result:
                result[predictor] = {}
            if target_subject not in result[predictor]:
                result[predictor][target_subject] = {}
            result[predictor][target_subject][target_roi] = score

        return result

    def get_score(
        self,
        predictor_subject: str,
        predictor_roi: str,
        target_subject: str,
        target_roi: str,
        default: float = 0.0,
    ) -> float:
        """Get score for a specific predictor-target combination.

        Args:
            predictor_subject: Predictor subject identifier.
            predictor_roi: Predictor ROI name.
            target_subject: Target subject identifier.
            target_roi: Target ROI name.
            default: Default value if not found.

        Returns:
            Alignment score.
        """
        mask = (
            (self.df["predictor_subject"] == predictor_subject)
            & (self.df["predictor_roi"] == predictor_roi)
            & (self.df["target_subject"] == target_subject)
            & (self.df["target_roi"] == target_roi)
        )
        matches = self.df[mask]
        return float(matches["score"].iloc[0]) if len(matches) > 0 else default

    def get_scores_for_predictor_target(
        self, predictor_subject: str, predictor_roi: str, target_roi: str
    ) -> list[float]:
        """Get all scores for a predictor-target pair across all target subjects.

        Args:
            predictor_subject: Predictor subject identifier.
            predictor_roi: Predictor ROI name.
            target_roi: Target ROI name.

        Returns:
            List of scores across all target subjects.
        """
        mask = (
            (self.df["predictor_subject"] == predictor_subject)
            & (self.df["predictor_roi"] == predictor_roi)
            & (self.df["target_roi"] == target_roi)
        )
        return self.df[mask]["score"].tolist()

    def get_average_score(
        self, predictor_subject: str, predictor_roi: str, target_roi: str
    ) -> float:
        """Get average score for a predictor-target pair across all target subjects.

        Args:
            predictor_subject: Predictor subject identifier.
            predictor_roi: Predictor ROI name.
            target_roi: Target ROI name.

        Returns:
            Average score.
        """
        scores = self.get_scores_for_predictor_target(predictor_subject, predictor_roi, target_roi)
        return float(np.mean(scores)) if scores else 0.0

    def get_pattern_for_predictor_target(
        self,
        predictor_subject: str,
        predictor_roi: str,
        target_subject: str,
        target_rois: list[str],
    ) -> np.ndarray:
        """Get alignment pattern vector for a predictor-target pair.

        Args:
            predictor_subject: Predictor subject identifier.
            predictor_roi: Predictor ROI name.
            target_subject: Target subject identifier.
            target_rois: List of target ROIs in desired order.

        Returns:
            Array of scores for each target ROI.
        """
        mask = (
            (self.df["predictor_subject"] == predictor_subject)
            & (self.df["predictor_roi"] == predictor_roi)
            & (self.df["target_subject"] == target_subject)
        )
        subset = self.df[mask].set_index("target_roi")["score"]
        result = np.array([float(subset.get(roi, np.nan)) for roi in target_rois])
        return result

    def get_pattern_matrix(
        self, predictor_roi: str, target_rois: list[str]
    ) -> np.ndarray:
        """Get alignment pattern matrix for a predictor.

        Rows are target ROIs, columns are target subjects.

        Args:
            predictor_subject: Predictor subject identifier.
            predictor_roi: Predictor ROI name.
            target_rois: List of target ROIs in desired order.

        Returns:
            Array of shape (n_target_rois, n_target_subjects).
        """
        filtered_ap_ds = self.filter(predictor_rois=[predictor_roi], target_rois=target_rois)
        matrix = filtered_ap_ds.df.pivot(
            index="target_roi", columns=["target_subject", "predictor_subject"], values="score"
        ).reindex(target_rois).to_numpy()
        return matrix

    def add_score(
        self,
        predictor_subject: str,
        predictor_roi: str,
        target_subject: str,
        target_roi: str,
        score: float,
    ) -> None:
        """Add or update a score.

        Args:
            predictor_subject: Predictor subject identifier.
            predictor_roi: Predictor ROI name.
            target_subject: Target subject identifier.
            target_roi: Target ROI name.
            score: Alignment score.
        """
        mask = (
            (self.df["predictor_subject"] == predictor_subject)
            & (self.df["predictor_roi"] == predictor_roi)
            & (self.df["target_subject"] == target_subject)
            & (self.df["target_roi"] == target_roi)
        )
        if mask.any():
            self.df.loc[mask, "score"] = score
        else:
            new_row = pd.DataFrame([{
                "predictor_subject": predictor_subject,
                "predictor_roi": predictor_roi,
                "target_subject": target_subject,
                "target_roi": target_roi,
                "score": score,
            }])
            self.df = pd.concat([self.df, new_row], ignore_index=True)

    def get_predictor_subjects(self) -> list[str]:
        """Get list of all unique predictor subjects."""
        return sorted(self.df["predictor_subject"].unique().tolist())

    def get_predictor_rois(self) -> list[str]:
        """Get list of all unique predictor ROIs."""
        return sorted(self.df["predictor_roi"].unique().tolist())

    def get_target_subjects(self) -> list[str]:
        """Get list of all unique target subjects."""
        return sorted(self.df["target_subject"].unique().tolist())

    def get_target_rois(self) -> list[str]:
        """Get list of all unique target ROIs."""
        return sorted(self.df["target_roi"].unique().tolist())

    def filter(
        self,
        predictor_subjects: Optional[list[str]] = None,
        predictor_rois: Optional[list[str]] = None,
        target_subjects: Optional[list[str]] = None,
        target_rois: Optional[list[str]] = None,
    ) -> "AlignmentPatternData":
        """Filter data by predictor or target dimensions.

        Args:
            predictor_subjects: List of predictor subjects to include (None = all).
            predictor_rois: List of predictor ROIs to include (None = all).
            target_subjects: List of target subjects to include (None = all).
            target_rois: List of target ROIs to include (None = all).

        Returns:
            New AlignmentPatternData instance with filtered data.
        """
        df = self.df.copy()
        if predictor_subjects is not None:
            df = df[df["predictor_subject"].isin(predictor_subjects)]
        if predictor_rois is not None:
            df = df[df["predictor_roi"].isin(predictor_rois)]
        if target_subjects is not None:
            df = df[df["target_subject"].isin(target_subjects)]
        if target_rois is not None:
            df = df[df["target_roi"].isin(target_rois)]
        return AlignmentPatternData(df)
