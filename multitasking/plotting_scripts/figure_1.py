#!/usr/bin/env python3
"""Generate figure 1 for boldmoments analysis.

For a given ROI x metric x main_outcome, figure 1 consists of 2 panels:
1. Alignment patterns: brain-brain and model-brain alignment, for every model
   that is statistically equivalent to the best model on the main outcome or
   on the alignment-pattern similarity (APS).
2. Score vs. APS: scatter plot with equivalence bands

By default the taskonomy models take part in the alignment-pattern panel on
the same terms as every other model: their pattern is drawn when they clear
the equivalence bar, and not otherwise. Pass ``--taskonomy False`` to exclude
them from that selection, which reproduces the behaviour this script had
before the option existed. They are in the scatter panel and in the
equivalence bands either way, since the best model on either axis is picked
over all models upstream and may well be a taskonomy model.
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
from matplotlib.ticker import FuncFormatter, MaxNLocator

from multitasking.analyses.alignment_patterns.base import (
    SIMILARITY_METRICS,
    aggregation_pickle,
)
from multitasking.analyses.dataframe_utils import get_order
from multitasking.CONSTANTS import get_rois, get_subjects, parse_roi_names
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


def set_aps_x_ticks(ax, step: float = 0.4, max_ticks: int = 6) -> None:
    """Puts fixed x ticks every ``step`` on the APS axis, when that is legible.

    :func:`set_ticks_0_1` lays ticks from the left limit up to 1 with no regard
    for how many that is, which suits an axis of Pearson correlations: those
    span about 1.2, so ``step=0.4`` yields three or four ticks. The APS axis is
    whatever the similarity metric produces, and a negated-MSE axis is not on
    that scale -- it can span 8 (twenty ticks, a black smear of overlapping
    labels) or 0.3 (one tick). Outside the legible band the fixed ticks are
    dropped for matplotlib's adaptive locator, capped at a few ticks and
    formatted to three significant digits so the labels stay short.

    Args:
        ax: Axis whose x ticks to set.
        step: Spacing between fixed ticks.
        max_ticks: Most fixed ticks to accept before falling back.
    """
    xlim = ax.get_xlim()
    ticks = np.arange(xlim[0], 1, step)
    n_visible = int(((ticks >= xlim[0]) & (ticks <= xlim[1])).sum())
    if 2 <= n_visible <= max_ticks:
        set_ticks_0_1(ax, start=xlim[0], axis="x", step=step)
    else:
        logger.info(
            f"APS axis spans {xlim[1] - xlim[0]:.4g}: ticks every {step} would "
            f"give {n_visible}, outside [2, {max_ticks}]. Using the adaptive "
            f"locator instead."
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3g}"))


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
    "--taskonomy",
    type=bool,
    default=True,
    show_default=True,
    help=(
        "Whether the taskonomy models can appear in the alignment-pattern "
        "panel, i.e. whether their pattern is drawn when they are equivalent "
        "to the best model. They are in the scatter panel and in the "
        "equivalence bands regardless of this flag."
    ),
)
@click.option(
    "--plot_path",
    type=click.Path(exists=False, file_okay=False, path_type=Path),
    default=None,
    help="Output directory for plots (default: results/boldmoments/plots)",
)
@click.option(
    "--dataset",
    type=click.Choice(["bold_moments", "nsd", "nsd_synthetic"]),
    default="bold_moments",
    show_default=True,
    help="Which dataset's canonical ROI and subject lists to use.",
)
@click.option(
    "--exclude-rois",
    type=str,
    default="",
    help=(
        "Comma-separated ROI names to drop entirely, e.g. 'V6,V6A'. Must "
        "match what aggregate_results was run with, since the alignment "
        "patterns in the pickles are vectors over the retained ROIs."
    ),
)
@click.option(
    "--aps-similarity-metric",
    type=click.Choice(list(SIMILARITY_METRICS)),
    default="pearson",
    show_default=True,
    help=(
        "Alignment-pattern similarity metric aggregate_results was run with. "
        "It is part of every pickle name, so this selects which aggregation "
        "in $SCRATCH to read."
    ),
)
def main(dataset, metric, roi, main_outcome, split, taskonomy, plot_path,
         exclude_rois, aps_similarity_metric):
    rois = get_rois(dataset, exclude=parse_roi_names(exclude_rois))
    subjects = get_subjects(dataset)
    date_str = datetime.now().strftime("%Y%m%d")
    # Paths
    scratch = Path(os.environ.get("SCRATCH", "/scratch"))
    project_root = Path(__file__).resolve().parent.parent.parent

    if plot_path is None:
        plot_path = project_root / "results" / "boldmoments" / "plots" / date_str
    plot_path.mkdir(parents=True, exist_ok=True)

    # Load data
    subject_avg_df = pickle.load(
        open(
            aggregation_pickle(
                scratch, f"subject_avg_df_equiv_{main_outcome}", aps_similarity_metric
            ),
            "rb",
        )
    )
    pairwise_subject_df = pickle.load(
        open(
            aggregation_pickle(
                scratch, "pairwise_subject_df", aps_similarity_metric
            ),
            "rb",
        )
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

    # Filter data for panels. ``all_models_df`` keeps the taskonomy models and
    # is what the scatter panel and the equivalence bands are built from: the
    # best model on the main outcome and on APS is picked over all models in
    # aggregate_results, so it may well be a taskonomy model and has to be
    # present here. ``panel_1_df`` is the subset whose alignment patterns are
    # drawn, which is where --taskonomy applies.
    all_models_df = subject_avg_df[
        (subject_avg_df["split"] == split)
        & (subject_avg_df["metric"] == metric)
        & (subject_avg_df["is_best_layer"])
    ]
    if taskonomy:
        panel_1_df = all_models_df
    else:
        panel_1_df = all_models_df[all_models_df["model_provider"] != "taskonomy"]

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
            aggregation_pickle(
                scratch, f"model_brain_alignment_patterns_{metric}_{main_outcome}", aps_similarity_metric
            ),
            "rb",
        )
    )
    brain_brain_ap = pickle.load(
        open(
            aggregation_pickle(
                scratch, f"pw_brain_brain_alignment_patterns_{metric}_{main_outcome}", aps_similarity_metric
            ),
            "rb",
        )
    )
    brain_brain_sim_df = brain_brain_ap.get_brain_brain_alignment_pattern_similarity(
        rois=rois,
        subjects=subjects,  # without these the defaults are BOLD Moments', and
                            # another dataset's subjects all miss -> NaN similarity
        # same similarity metric as the model-brain alignment patterns
        similarity_metric=model_brain_ap.similarity_metric,
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

    # The best model on each axis is drawn whether or not it clears the
    # equivalence bar. For the main outcome that is already guaranteed -- the
    # best model's mean lies inside its own bootstrap CI, so it is always in
    # ``relevant_predictors``. For the APS axis it is not: a model can top the
    # alignment-pattern similarity while scoring well below the best model,
    # which puts it in ``aps_predictors`` only, and the loop below draws
    # nothing for those. Without this it vanishes from the panel -- on NSD
    # under MSE that was 26 of 38 (ROI x metric) panels.
    best_predictors = set(
        roi_df.query("is_best_model or is_best_model_aps")["predictor"].unique()
    )

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
        subjects) # average target patterns by ROI
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

    # dict.fromkeys de-duplicates while keeping the order: a predictor that is
    # equivalent on both axes appears in both lists and would otherwise be
    # drawn (and legended) twice.
    all_predictors = list(
        dict.fromkeys(np.concatenate([relevant_predictors, aps_predictors]))
    )

    seen_labels: set[str] = set()

    def legend_label(model: str) -> str:
        """Returns the legend entry for a model, once per distinct entry.

        All taskonomy models share one colour, so they share one entry too;
        repeats become "_nolegend_" so the whole family does not flood the
        legend.

        Args:
            model: Model name, e.g. ``"taskonomy/normal"``.

        Returns:
            The legend label, or ``"_nolegend_"`` if it was already used.
        """
        label = (
            "taskonomy"
            if model.startswith("taskonomy/")
            else model_legend_mapping[model]
        )
        if label in seen_labels:
            return "_nolegend_"
        seen_labels.add(label)
        return label

    n_taskonomy = sum(p.startswith("taskonomy/") for p in all_predictors)
    logger.info(
        f"Plotting {len(all_predictors)} equivalent predictor alignment "
        f"patterns ({n_taskonomy} taskonomy)"
    )
    for predictor in all_predictors:
        model = predictor.split("__")[0]
        # legend_label spends an entry the first time it is asked for one, so
        # it is called inside the branches: asking for a predictor that is
        # then not drawn would suppress that model's later legend entry.
        if (predictor in aps_predictors) and (predictor in relevant_predictors):
            axes[0].plot(
                range(len(rois)),
                mb_aps[predictor].mean(axis=0),
                color=model_colors_dict[model],
                label=legend_label(model),
                zorder=0, alpha=1, linestyle="-.",
            )
        elif predictor in relevant_predictors:
            axes[0].plot(
                range(len(rois)),
                mb_aps[predictor].mean(axis=0),
                color=model_colors_dict[model],
                linestyle="--",
                label=legend_label(model),
                alpha=0.3, zorder=0,
            )
        elif predictor in best_predictors:
            # Best on APS but not equivalent on the main outcome. Dash-dot for
            # "selected on APS", faint for "not score-equivalent", matching
            # how the two branches above use linestyle and alpha.
            axes[0].plot(
                range(len(rois)),
                mb_aps[predictor].mean(axis=0),
                color=model_colors_dict[model],
                linestyle="-.",
                label=legend_label(model),
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
        data=all_models_df.query("roi == @roi"),
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
    tmp = all_models_df.query(
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

    tmp = all_models_df.query("roi == @roi and is_best_model_aps")
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
    set_aps_x_ticks(axes[1], step=0.4)

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
    fname = f"figure_1_{roi}_{metric}_{main_outcome}_{taskonomy}"
    save_this(
        plot_path,
        fname,
        formats=["png"],
        transparent=False,
        dpi=300,
    )
    logger.info(f"Saved figure to {plot_path}/{fname}.*")


if __name__ == "__main__":
    main()
