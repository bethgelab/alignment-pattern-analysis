"""Equivalence analysis."""

import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def run_equivalence_analysis(full_df, main_outcome="score"):
    group_cols = ["roi", "metric"]
    test_best_layer = (full_df["split"] == "test") & (full_df["is_best_layer"])

    # 1️⃣ Extract best-on-score rows (score CI only)
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

    # 2️⃣ Extract best-on-APS rows (APS CI only)
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

    # 3️⃣ Merge best CIs onto all models
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

    # 4️⃣ Score equivalence (vs best model on score)
    full_df["is_equivalent_to_best_on_score"] = (
        full_df[f"{main_outcome}_mean"] >= full_df["best_score_ci_lower"]
    )
    logger.info(f"Full df after scoring equivalence: {full_df.shape}")

    # 5️⃣ APS equivalence (vs best model on APS)
    full_df["is_equivalent_to_best_on_aps"] = (
        full_df["aps_score_mean"] >= full_df["best_aps_ci_lower"]
    )
    logger.info(f"Full df after APS equivalence: {full_df.shape}")
    return full_df


def collect_equivalent_models(
    df, roi, models, boot_cis, metric, min_score=None
):
    best_model_roi = df["model"].iloc[
        df[f"model_order_{metric}_{roi}"].cat.codes.idxmax()
    ]
    lower_of_best, _ = boot_cis[metric][roi][best_model_roi]

    # Count of overlapping models for this ROI and metric
    n_overlap = 0

    equivalent_models = []
    equivalent_models_layers = []
    equivalent_models_indices = []
    for model_index, model in enumerate(models):
        # Get the avg score for the model on the ROI
        (m, layer) = df[
            (df["roi"] == roi)
            & (df["model"] == model)
            & (df["split"] == "test")
            & (df["metric"] == metric)
        ][["score_mean", "layer"]].to_numpy()[0]
        if m >= lower_of_best and (min_score is None or m >= min_score):
            # Use model name and ROI name as indices instead of numeric indices
            equivalent_models.append(model)
            equivalent_models_layers.append(layer)
            n_overlap += 1
            equivalent_models_indices.append(model_index)
    return equivalent_models, equivalent_models_layers, equivalent_models_indices, n_overlap


def ci_plots(df,
             roi,
             boot_cis,
             metric,
             models,
             ax,
             xoffset=0,
             metric_color_dict=None):
    best_model_roi = df['model'].iloc[df[f"model_order_{metric}_{roi}"].cat.codes.idxmax()]
    lower_of_best, _ = boot_cis[metric][roi][best_model_roi]
    plt.sca(ax)
    for model_index, model in enumerate(models):
        m = df[(df['roi']==roi) &
               (df['model']==model) &
               (df['split']=='test') &
               (df['metric']==metric)
               ]['score_mean'].to_numpy()
        ci_lower, ci_upper = boot_cis[metric][roi][model]
        if m >= lower_of_best:
            marker = "*"
            color=metric_color_dict[metric]
            alpha=1
        else:
            marker="o"
            color=metric_color_dict[metric]
            alpha=.3
        plt.scatter(
            model_index+xoffset,
                     y= m,
                     color=color,
                     alpha=alpha,
                     marker=marker)
        plt.errorbar(
            model_index+xoffset,
                     y= (ci_upper + ci_lower)/2,
                     yerr=(ci_upper - ci_lower)/2,
                     color=metric_color_dict[metric],
                    alpha=alpha,
                     zorder=0
                     )
        if model == best_model_roi:
            plt.hlines(y=ci_lower,
                       xmin=0,
                       xmax=len(models),
                       color=metric_color_dict[metric],
                       zorder=0)
    return ax
