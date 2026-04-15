#!/usr/bin/env python3
"""Generate figure 1 for boldmoments analysis.

For a given ROI x metric x main_outcome, figure 1 consists of 3 panels:
1. Model ranking: benchmarking results as bar plot with CIs
2. Alignment patterns: brain-brain and model-brain alignment
3. Score vs. APS: scatter plot with equivalence bands
"""

import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from multitasking.analyses.dataframe_utils import get_order
from multitasking.CONSTANTS import ROIS as rois
from multitasking.CONSTANTS import SUBJECTS
from multitasking.plot_creation import cmap_colors
from multitasking.plot_creation.plot_utils import (
    save_this,
    set_ticks_0_1,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--metric",
    type=str,
    default="rsa",
    help="Metric to use (default: rsa)",
)
@click.option(
    "--roi",
    "roi",
    type=str,
    default="V6",
    help="ROI to plot (default: V6)",
)
@click.option(
    "--main-outcome",
    type=click.Choice(["score", "score_norm_upper", "score_norm_lower"]),
    default="score",
    help="Main outcome variable (default: score)",
)
@click.option(
    "--split",
    type=str,
    default="test",
    help="Data split (default: test)",
)
@click.option(
    "--plot_path",
    type=click.Path(exists=False, file_okay=False, path_type=Path),
    default=None,
    help="Output directory for plots (default: results/boldmoments/plots)",
)
def main(metric, roi, main_outcome, split, plot_path):
    date_str = datetime.now().strftime("%Y%m%d")
    # Paths
    scratch = Path(os.environ.get("SCRATCH", "/scratch"))
    project_root = Path(__file__).resolve().parent.parent.parent

    if plot_path is None:
        plot_path = project_root / "results" / "boldmoments" / "plots" / date_str
    plot_path.mkdir(parents=True, exist_ok=True)

    # Load data
    subject_avg_df = pickle.load(
        open(scratch / f"subject_avg_df_equiv_{main_outcome}.pkl", "rb")
    )
    pairwise_subject_df = pickle.load(
        open(scratch / "pairwise_subject_df.pkl", "rb")
    )

    subject_avg_df["model_provider"] = subject_avg_df["model"].str.split("/").str[0]
    subject_avg_df["predictor"] = (
        subject_avg_df["model"].astype(str)
        + "__"
        + subject_avg_df["layer"].astype(str)
    )

    # Model colors
    with open(project_root / "indiv_model_colors_dict.json") as f:
        model_colors_dict = json.load(f)

    # Model legend mapping
    with open(project_root / "model_legend_mapping.json") as f:
        model_legend_mapping = json.load(f)

    # Filter data for panels
    panel_1_df = subject_avg_df[
        (subject_avg_df["split"] == split)
        & (subject_avg_df["metric"] == metric)
        & (subject_avg_df["is_best_layer"])
        & (subject_avg_df["model_provider"] != "taskonomy")
    ]
    panel_1_df_taskonomy = subject_avg_df[
        (subject_avg_df["split"] == split)
        & (subject_avg_df["metric"] == metric)
        & (subject_avg_df["is_best_layer"])
    ]

    roi_df = panel_1_df.query(f"roi == '{roi}'")
    model_order = get_order(
        panel_1_df,
        column_name=f"{main_outcome}_mean",
        metric=metric,
        order_by="model",
    )
    roi_df = roi_df.set_index("model").loc[model_order].reset_index()

    logger.info(f"Loading model-brain alignment patterns for {metric} and {main_outcome}...")
    # Load alignment patterns
    model_brain_ap = pickle.load(
        open(
            scratch
            / f"model_brain_alignment_patterns_{metric}_{main_outcome}.pkl",
            "rb",
        )
    )
    brain_brain_ap = pickle.load(
        open(
            scratch
            / f"pw_brain_brain_alignment_patterns_{metric}_{main_outcome}.pkl",
            "rb",
        )
    )
    brain_brain_sim_df = brain_brain_ap.get_brain_brain_alignment_pattern_similarity(
        similarity_metric=model_brain_ap.similarity_metric # use same similarity metric as model-brain alignment patterns
    )

    brain_brain_sim = brain_brain_sim_df.query("roi == @roi")["similarity"].values
    brain_brain_sim_mean = brain_brain_sim.mean()
    n = len(brain_brain_sim)
    logger.info(f"Number of brain-brain similarity values: {n}")
    brain_brain_sim_sem = brain_brain_sim.std(ddof=1) / np.sqrt(n)

    roi_df["predictor"] = (
        roi_df["model"].astype(str) + "__" + roi_df["layer"].astype(str)
    )
    relevant_predictors = roi_df.query("is_equivalent_to_best_on_score")[
        "predictor"
    ].unique()
    aps_predictors = roi_df.query("is_equivalent_to_best_on_aps")[
        "predictor"
    ].unique()

    mb_aps = model_brain_ap.get_alignment_pattern_dict_by_predictor(
        np.concatenate([relevant_predictors, aps_predictors]), rois
    )

    roi_color_dict = {r: cmap_colors[i] for i, r in enumerate(rois)}

    # Create figure
    plt.rcParams.update({
        "figure.dpi": 300,
        "font.family": "serif",
        "axes.labelsize": 6,
        "axes.titlesize": 6,
        "legend.fontsize": 6,
        "legend.title_fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "font.size": 6,
        "lines.linewidth": 1,
        "lines.markersize": 3,
    })

    fig = plt.figure(figsize=(2.75, 1), constrained_layout=True)
    gs = fig.add_gridspec(
        1, 2, width_ratios=[2, 1], wspace=0
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    axes = [ax1, ax2]
    axes[1].tick_params(labelleft=False)
    [ax.yaxis.set_tick_params(pad=1) for ax in axes]

    # Panel 2: Alignment patterns
    bb_aps_roi = brain_brain_ap.get_avg_target_pattern_dict_by_roi(
        rois,
        SUBJECTS) # average target patterns by ROI
    brain_mean = bb_aps_roi[roi].mean(axis=0)
    brain_std = bb_aps_roi[roi].std(axis=0, ddof=1)
    n = bb_aps_roi[roi].shape[0]
    logger.info(f"Number of brain-brain similarity values: {n}")
    brain_sem = brain_std / np.sqrt(n)
    axes[0].fill_between(
        range(len(rois)),
        brain_mean - brain_sem,
        brain_mean + brain_sem,
        color=roi_color_dict[roi],
        alpha=0.3,
        edgecolor="none",
    )
    axes[0].plot(range(len(rois)), brain_mean, color=roi_color_dict[roi], zorder=100,
                 label=f"fMRI data ({roi})")

    axes[0].fill_betweenx(
        [0, 1],
        rois.index(roi) - 0.5,
        rois.index(roi) + 0.5,
        color="k",
        alpha=0.1,
        edgecolor="none",
        zorder=0,
    )

    all_predictors = np.concatenate([relevant_predictors, aps_predictors])
    for predictor in all_predictors:
        if (predictor in aps_predictors) and (predictor in relevant_predictors):
            axes[0].plot(
                range(len(rois)),
                mb_aps[predictor].mean(axis=0),
                color=model_colors_dict[predictor.split("__")[0]],
                label=model_legend_mapping[predictor.split("__")[0]],
                zorder=0, alpha=1, linestyle="-.",
            )
        elif predictor in relevant_predictors:
            axes[0].plot(
                range(len(rois)),
                mb_aps[predictor].mean(axis=0),
                color=model_colors_dict[predictor.split("__")[0]],
                linestyle="--",
                label=model_legend_mapping[predictor.split("__")[0]],
                alpha=0.3, zorder=0,
            )
        else:
            pass

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].set_xticks(range(len(rois)), rois, rotation=90)
    axes[0].set_xlabel("ROI")
    # axes[1].set_yticklabels([f"" for y in axes[1].get_yticks()])

    # Panel 3: Score vs. APS
    sns.scatterplot(
        data=panel_1_df_taskonomy.query("roi == @roi"),
        x="aps_score_mean",
        y=f"{main_outcome}_mean",
        hue="model",
        palette=model_colors_dict,
        ax=axes[1],
        legend=False,
    )

    noise_ceiling_mean = pairwise_subject_df.query(
        "roi == @roi and layer == @roi and metric == @metric and split == @split"
    )[main_outcome].mean()
    noise_ceiling_std = pairwise_subject_df.query(
        "roi == @roi and layer == @roi and metric == @metric and split == @split"
    )[main_outcome].std(ddof=1)
    n = pairwise_subject_df.query(
        "roi == @roi and layer == @roi and metric == @metric and split == @split"
    )[main_outcome].shape[0]
    noise_ceiling_sem = noise_ceiling_std / np.sqrt(n)
    tmp = panel_1_df_taskonomy.query(
            "roi == @roi and is_best_layer and is_best_model"
        )
    upper_ylim = max(noise_ceiling_mean + noise_ceiling_sem,tmp["best_score_ci_upper"].values[0])
    upper_xlim = brain_brain_sim_mean + brain_brain_sim_sem

    xlim = axes[1].get_xlim()
    ylim = axes[1].get_ylim()

    axes[1].set_xlim(xlim[0], upper_xlim)
    axes[1].set_ylim(ylim[0], upper_ylim)


    # ylim = axes[1].get_ylim()
    axes[1].fill_betweenx(
        ylim,
        tmp["best_aps_ci_lower"].values[0],
        tmp["best_aps_ci_upper"].values[0],
        color="k",
        alpha=0.1,
        linewidth=0,
    )

    tmp = panel_1_df_taskonomy.query("roi == @roi and is_best_model_aps")
    axes[1].fill_between(
        [xlim[0], upper_xlim],
        tmp["best_score_ci_lower"].values[0],
        tmp["best_score_ci_upper"].values[0],
        color="k",
        alpha=0.1,
        linewidth=0,
    )

    axes[1].vlines(
        x=brain_brain_sim_mean,
        ymin=ylim[0],
        ymax=ylim[1],
        color="k",
        alpha=1,
        linewidth=1,
        linestyle="-.",
    )
    logger.info(f"Brain-brain similarity mean: {brain_brain_sim_mean}, sem: {brain_brain_sim_sem}")
    axes[1].fill_betweenx(
        ylim,
        brain_brain_sim_mean - brain_brain_sim_sem,
        brain_brain_sim_mean + brain_brain_sim_sem,
        color="k",
        alpha=0.2,
        linewidth=0,
    )


    # bbox = axes[2].get_position()
    # fig.legend(
    #     handles,
    #     labels,
    #     title="Predictor feature spaces",
    #     title_fontsize=8,
    #     loc="upper center",
    #     bbox_to_anchor=(bbox.x0 + bbox.width / 2, bbox.y0 - 0.15),
    #     bbox_transform=fig.transFigure,
    #     fontsize=8,
    #     frameon=False,
    #     ncol=2,
    # )

    sns.despine()

    set_ticks_0_1(axes[0], axis="y", step=0.4)
    set_ticks_0_1(axes[1], start=axes[1].get_xlim()[0], axis="x", step=0.4)

    # axes[0].set_title("Model ranking")
    # axes[1].set_title("Alignment patterns")
    # axes[2].set_title("Score vs. APS")

    axes[0].set_ylabel("")
    axes[0].set_xlabel("")
    # axes[1].set_ylabel(f" avg. {main_outcome}")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    for ax in axes:
        ax.tick_params(axis="y",
                       labelsize=plt.rcParams["ytick.labelsize"],
                       which="major",
                       pad=1)
        ax.tick_params(axis="x",
                       labelsize=plt.rcParams["xtick.labelsize"],
                       pad=1)

    global_ymax = -np.inf
    for ax in axes:
        for line in ax.get_lines():
            y = line.get_ydata()
            y = y[np.isfinite(y)]
            if len(y):
                global_ymax = max(global_ymax, y.max())

    axes[0].set_ylim(0, global_ymax+0.05)

    # plt.suptitle(f"ROI: {roi}, metric: {metric}")
    save_this(
        plot_path,
        f"figure_1_{roi}_{metric}_{main_outcome}",
        formats=["png", "svg"],
        transparent=False,
        dpi=300,
    )
    logger.info(f"Saved figure to {plot_path}/figure_1_{roi}_{metric}_{main_outcome}.*")


if __name__ == "__main__":
    main()
