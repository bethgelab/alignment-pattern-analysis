"""Equivalence analysis."""

import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def run_equivalence_analysis(full_df, main_outcome="score"):
    group_cols = ["roi", "metric"]
    test_best_layer = (full_df["split"] == "test") & (full_df["is_best_layer"])

    # Extract best-on-score rows (score CI only)
    best_score_df = (
        full_df.loc[(full_df["is_best_model"]) & test_best_layer,
                    group_cols + [f"{main_outcome}_ci"]]
        .copy()
    )
    best_score_df["best_score_ci_lower"] = best_score_df[f"{main_outcome}_ci"].str[0]
    best_score_df["best_score_ci_upper"] = best_score_df[f"{main_outcome}_ci"].str[1]
    best_score_df = best_score_df[group_cols + ["best_score_ci_lower", "best_score_ci_upper"]]
    logger.info(f"Best-on-score rows: {best_score_df.shape}")

    assert (
        full_df[(full_df["is_best_model"]) & test_best_layer]
        .groupby(group_cols)
        .size()
        .eq(1)
        .all()
    ), "Expected exactly one best model on score per (roi, metric)"

    # Extract best-on-APS rows (APS CI only)
    best_aps_df = (
        full_df.loc[(full_df["is_best_model_aps"]) & test_best_layer,
                    group_cols + ["aps_score_ci"]]
        .copy()
    )
    best_aps_df["best_aps_ci_lower"] = best_aps_df["aps_score_ci"].str[0]
    best_aps_df["best_aps_ci_upper"] = best_aps_df["aps_score_ci"].str[1]
    best_aps_df = best_aps_df[group_cols + ["best_aps_ci_lower", "best_aps_ci_upper"]]
    logger.info(f"Best-on-APS rows: {best_aps_df.shape}")

    assert (
        full_df[(full_df["is_best_model_aps"]) & test_best_layer]
        .groupby(group_cols)
        .size()
        .eq(1)
        .all()
    ), "Expected exactly one best model on APS per (roi, metric)"

    # Merge best CIs onto all models
    full_df = full_df.merge(
        best_score_df,
        on=group_cols,
        how="left",
        validate="m:1",
    )
    full_df = full_df.merge(
        best_aps_df,
        on=group_cols,
        how="left",
        validate="m:1",
    )
    logger.info(f"Full df after merging best CIs: {full_df.shape}")

    # Score equivalence (vs best model on score)
    full_df["is_equivalent_to_best_on_score"] = (
        full_df[f"{main_outcome}_mean"] >= full_df["best_score_ci_lower"]
    )
    logger.info(f"Full df after scoring equivalence: {full_df.shape}")

    # APS equivalence (vs best model on APS)
    full_df["is_equivalent_to_best_on_aps"] = (
        full_df["aps_score_mean"] >= full_df["best_aps_ci_lower"]
    )
    logger.info(f"Full df after APS equivalence: {full_df.shape}")
    return full_df
