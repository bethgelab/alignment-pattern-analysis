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

from multitasking.CONSTANTS import ROIS as rois
from multitasking.plot_creation.dataframe_utils import get_order
from multitasking.plot_creation.plot_utils import (
    add_noise_ceiling,
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
    models_short_path = \
        Path("/mnt/lustre/work/bethge/bkr578/projects/multitasking/results/boldmoments/models_shortened.txt")

    if plot_path is None:
        plot_path = Path("/mnt/lustre/work/bethge/bkr578/projects/multitasking/results/boldmoments/plots") / f"{date_str}"
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
    model_colors_dict_by_family = pickle.load(
        open(scratch / "model_colors_dict.pkl", "rb")
    )
    model_colors_dict = {}
    for model_provider in subject_avg_df["model_provider"].unique():
        models = subject_avg_df.query(
            "model_provider == @model_provider and is_best_layer"
        )["model"].unique()
        if model_provider != "taskonomy":
            model_colors_dict.update(
                {
                    model: model_colors_dict_by_family[model_provider][i]
                    for i, model in enumerate(models)
                }
            )
        else:
            model_colors_dict.update(
                {model: "gray" for i, model in enumerate(models)}
            )

    # Model legend mapping
    models = subject_avg_df["model"].unique().tolist()
    with open(models_short_path) as f:
        models_short = json.load(f)
    model_legend_mapping = dict(zip(models, models_short, strict=False))

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

    means = roi_df[f"{main_outcome}_mean"].values
    cis = roi_df[f"{main_outcome}_ci"].values
    is_equivalent = roi_df["is_equivalent_to_best_on_score"].values

    lower_errors = means - np.array([ci[0] for ci in cis])
    upper_errors = np.array([ci[1] for ci in cis]) - means
    yerrs = np.vstack([lower_errors, upper_errors])
    x = np.arange(len(model_order))

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

    bb_aps = brain_brain_ap.get_alignment_pattern_dict_by_roi(rois)
    brain_brain_sim_df = brain_brain_ap.get_brain_brain_alignment_pattern_similarity(
        similarity_metric=model_brain_ap.similarity_metric # use same similarity metric as model-brain alignment patterns
    )

    brain_brain_sim = brain_brain_sim_df.query("roi == @roi")["similarity"].values
    brain_brain_sim_mean = brain_brain_sim.mean()
    brain_brain_sim_sem = brain_brain_sim.std() / np.sqrt(len(brain_brain_sim))

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

    # Create figure
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "font.size": 8,
            "lines.linewidth": 1,
            "lines.markersize": 3,
        }
    )

    fig = plt.figure(figsize=(5.5, 1), constrained_layout=True)
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.2, 1, 0.5], wspace=0.1
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
    axes = [ax1, ax2, ax3]
    axes[1].tick_params(labelleft=False)
    axes[2].tick_params(labelleft=False)


    # Panel 1: Model ranking
    for i, mean, is_eq, yerr in zip(
        x[::-1], means[::-1], is_equivalent[::-1], yerrs.T[::-1], strict=False
    ):
        bars = axes[0].bar(
            i,
            mean,
            color=model_colors_dict[model_order[i]],
            alpha=1 if is_eq else 0.3,
        )
        axes[0].errorbar(
            i,
            mean,
            yerr=yerr[:, None],
            fmt="none",
            ecolor="k",
            alpha=1 if is_eq else 0.5,
            capsize=0,
        )

    bar_width = bars[0].get_width()
    left = x[0] - bar_width / 2
    right = x[-1] + bar_width / 2

    for noise_ceiling_df, ls in zip(
        [pairwise_subject_df], ["-."], strict=False
    ):
        add_noise_ceiling(
            axes[0],
            noise_ceiling_df[noise_ceiling_df["layer"] == roi],
            roi,
            split,
            metric,
            var_type="sem",
            color="k",
            linestyle=ls,
            xmin=left,
            xmax=right,
        )

    axes[0].set_xticks(
        x, [model_legend_mapping[model].replace("_", "-") for model in model_order], rotation=90
    )
    axes[0].set_ylabel(main_outcome, labelpad=2)

    # Panel 2: Alignment patterns
    brain_ap_matrix = bb_aps[roi]
    brain_mean = brain_ap_matrix.mean(axis=0)
    brain_std = brain_ap_matrix.std(axis=0)
    # logger.info(f"Brain STD: {brain_std}")
    brain_sem = brain_std / np.sqrt(brain_ap_matrix.shape[0])
    # logger.info(f"Brain SEM: {brain_sem}")
    axes[1].fill_between(
        range(len(rois)),
        brain_mean - brain_sem,
        brain_mean + brain_sem,
        color="k",
        alpha=0.1,
        edgecolor="none",
    )
    axes[1].plot(range(len(rois)), brain_mean, color="k",
                 label=f"fMRI data ({roi})")

    axes[1].fill_betweenx(
        axes[1].get_ylim(),
        rois.index(roi) - 0.5,
        rois.index(roi) + 0.5,
        color="k",
        alpha=0.1,
        edgecolor="none",
        zorder=0,
    )

    for predictor in relevant_predictors:
        axes[1].plot(
            range(len(rois)),
            mb_aps[predictor].mean(axis=0),
            color=model_colors_dict[predictor.split("__")[0]],
            # linestyle="--",
            label=model_legend_mapping[predictor.split("__")[0]],
        )

    for predictor in set(aps_predictors) - set(relevant_predictors):
        axes[1].plot(
            range(len(rois)),
            mb_aps[predictor].mean(axis=0),
            color=model_colors_dict[predictor.split("__")[0]],
            linestyle="--",
            label=model_legend_mapping[predictor.split("__")[0]],
            alpha= 0.3
        )

    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].set_xticks(range(len(rois)), rois, rotation=90)
    axes[1].set_xlabel("ROI")
    # axes[1].set_yticklabels([f"" for y in axes[1].get_yticks()])

    # Panel 3: Score vs. APS
    sns.scatterplot(
        data=panel_1_df_taskonomy.query("roi == @roi"),
        x="aps_score_mean",
        y=f"{main_outcome}_mean",
        hue="model",
        palette=model_colors_dict,
        ax=axes[2],
        legend=False,
    )

    noise_ceiling_mean = pairwise_subject_df.query(
        "roi == @roi and layer == @roi and metric == @metric and split == @split"
    )[main_outcome].mean()
    noise_ceiling_std = pairwise_subject_df.query(
        "roi == @roi and layer == @roi and metric == @metric and split == @split"
    )[main_outcome].std()
    n = pairwise_subject_df.query(
        "roi == @roi and layer == @roi and metric == @metric and split == @split"
    )[main_outcome].shape[0]
    noise_ceiling_sem = noise_ceiling_std / np.sqrt(n)
    tmp = panel_1_df_taskonomy.query(
            "roi == @roi and is_best_layer and is_best_model"
        )
    upper_ylim = noise_ceiling_mean + noise_ceiling_sem
    upper_xlim = brain_brain_sim_mean + brain_brain_sim_sem

    xlim = axes[2].get_xlim()
    ylim = axes[2].get_ylim()

    axes[2].set_xlim(xlim[0], upper_xlim)
    axes[2].set_ylim(ylim[0], upper_ylim)


    ylim = axes[2].get_ylim()
    axes[2].fill_betweenx(
        ylim,
        tmp["best_aps_ci_lower"].values,
        tmp["best_aps_ci_upper"].values,
        color="k",
        alpha=0.1,
        linewidth=0,
    )

    tmp = panel_1_df_taskonomy.query("roi == @roi and is_best_model_aps")
    axes[2].fill_between(
        [xlim[0], upper_xlim],
        tmp["best_score_ci_lower"].values,
        tmp["best_score_ci_upper"].values,
        color="k",
        alpha=0.1,
        linewidth=0,
    )

    axes[2].vlines(
        x=brain_brain_sim_mean,
        ymin=ylim[0],
        ymax=ylim[1],
        color="k",
        alpha=1,
        linewidth=1,
        linestyle="-.",
    )
    axes[2].fill_betweenx(
        ylim,
        brain_brain_sim_mean - brain_brain_sim_sem,
        brain_brain_sim_mean + brain_brain_sim_sem,
        color="k",
        alpha=0.1,
        linewidth=0,
    )

    add_noise_ceiling(
            axes[2],
            pairwise_subject_df[pairwise_subject_df["layer"] == roi],
            roi,
            split,
            metric,
            var_type="sem",
            color="k",
            linestyle="-.",
            xmin=left,
            xmax=right,
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

    set_ticks_0_1(axes[0], axis="y")
    set_ticks_0_1(axes[2], start=axes[2].get_xlim()[0], axis="x", step=0.2)

    # axes[0].set_title("Model ranking")
    # axes[1].set_title("Alignment patterns")
    # axes[2].set_title("Score vs. APS")

    axes[0].set_ylabel(f"Avg. {main_outcome}")
    # axes[1].set_ylabel(f" avg. {main_outcome}")
    axes[1].set_xlabel("")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("")

    for ax in axes:
        ax.tick_params(axis="y",
                       labelsize=plt.rcParams["ytick.labelsize"],
                       which="major",
                       pad=2)
        ax.tick_params(axis="x",
                       labelsize=plt.rcParams["xtick.labelsize"])

    global_ymax = -np.inf
    for ax in axes:
        for line in ax.get_lines():
            y = line.get_ydata()
            y = y[np.isfinite(y)]
            if len(y):
                global_ymax = max(global_ymax, y.max())

    ax1.set_ylim(0, global_ymax+0.05)

    # plt.suptitle(f"ROI: {roi}, metric: {metric}")
    save_this(
        plot_path,
        f"figure_1_{roi}_{metric}_{main_outcome}",
        formats=None,
        transparent=False,
        dpi=300,
    )
    logger.info(f"Saved figure to {plot_path}/figure_1_{roi}_{metric}_{main_outcome}.*")


if __name__ == "__main__":
    main()
