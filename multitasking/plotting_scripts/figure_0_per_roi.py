#!/usr/bin/env python3
"""Generate the model benchmark figure separately for each ROI.

``figure_0`` draws one bar per model whose height is the mean over ROIs and
whose error bar is the spread *across ROIs*, so it answers "which model is
best overall" and hides every regional difference. This script draws the same
bar chart per ROI instead: bar height is the model's subject-averaged score in
that one ROI, and the error bar is the by-subject bootstrap CI that
``aggregate_results`` already computed and that the equivalence analysis is
based on.

Because the CI is the same quantity the equivalence test uses, a panel can
also say which models are statistically indistinguishable from the best model
in that ROI: those bars are drawn with a black outline.

``figure_0`` is untouched -- it produced the published figures.
"""

import json
import logging
import math
import os
import pickle
from datetime import datetime
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from multitasking.analyses.dataframe_utils import (
    align_order_to_available,
    get_order,
)
from multitasking.CONSTANTS import get_rois, parse_roi_names
from multitasking.plot_creation.plot_utils import save_this

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Panel geometry, in inches. Single-ROI figures match figure_0's width.
PANEL_WIDTH = 2.75
PANEL_WIDTH_TASKONOMY = 5.5
PANEL_HEIGHT = 1.0
GRID_COLUMNS = 4
# Vertical room a rotated column of model names needs, in inches. Only the
# lowest grid panel of each column carries them.
GRID_LABEL_BAND = 1.6


def load_subject_avg_df(
    scratch: Path, main_outcome: str, split: str, metric: str, taskonomy: bool
) -> pd.DataFrame:
    """Load the aggregated per-(model, ROI) scores for one metric and outcome.

    Args:
        scratch: Directory ``aggregate_results`` wrote its pickles to.
        main_outcome: ``score``, ``score_norm_upper`` or ``score_norm_lower``.
        split: Data split to keep, normally ``test``.
        metric: ``rsa`` or ``linear_predictivity``.
        taskonomy: Whether to keep the taskonomy models.

    Returns:
        One row per (model, ROI), restricted to each model's best layer.
    """
    path = scratch / f"subject_avg_df_equiv_{main_outcome}.pkl"
    with open(path, "rb") as handle:
        df = pickle.load(handle)

    df["model_provider"] = df["model"].str.split("/").str[0]
    keep = (
        (df["split"] == split)
        & (df["metric"] == metric)
        & (df["is_best_layer"])
    )
    if not taskonomy:
        keep &= df["model_provider"] != "taskonomy"
    return df[keep]


def order_pickle_dir(scratch: Path, model_order_from: Path | None) -> Path:
    """Returns the directory the model ranking should be read from.

    Args:
        scratch: This dataset's own aggregation directory.
        model_order_from: A reference aggregation, either its directory or the
            ``subject_avg_df_equiv_score_norm_lower.pkl`` inside it. None
            ranks on this dataset's own pickles.

    Returns:
        A directory holding ``subject_avg_df_equiv_*.pkl``.
    """
    if model_order_from is None:
        return scratch
    return model_order_from.parent if model_order_from.is_file() else model_order_from


def global_model_order(scratch: Path, split: str, taskonomy: bool) -> list[str]:
    """Rank models worst-to-best the way ``figure_0`` does.

    Ordering by noise-ceiling-normalized RSA, pooled over ROIs, keeps the bars
    in the same position in every panel, so panels can be read side by side.
    Pointing ``scratch`` at another dataset's aggregation extends that to
    panels of different datasets.

    Args:
        scratch: Directory holding the aggregated pickles.
        split: Data split to rank on.
        taskonomy: Whether to include the taskonomy models.

    Returns:
        Model names, lowest score first.
    """
    df = load_subject_avg_df(
        scratch, "score_norm_lower", split, "rsa", taskonomy
    )
    return get_order(
        df,
        column_name="score_norm_lower_mean",
        metric="rsa",
        order_by="model",
    )


def error_bar_bounds(rows: pd.DataFrame, main_outcome: str) -> np.ndarray:
    """Turn the stored bootstrap CI into matplotlib's ``yerr`` offsets.

    The CI column holds ``(lower, upper)`` tuples in absolute score units,
    while ``errorbar`` wants distances from the bar top. Rows whose CI is
    missing fall back to the standard error over subjects.

    Args:
        rows: One row per bar, already in bar order.
        main_outcome: The outcome being plotted.

    Returns:
        A ``(2, n_bars)`` array of non-negative offsets, lower then upper.
    """
    means = rows[f"{main_outcome}_mean"].to_numpy(dtype=float)
    ci_column = f"{main_outcome}_ci"

    sem = (
        rows[f"{main_outcome}_std"].to_numpy(dtype=float)
        / np.sqrt(rows["count"].to_numpy(dtype=float))
    )
    lower = sem.copy()
    upper = sem.copy()

    if ci_column in rows.columns:
        for i, value in enumerate(rows[ci_column].to_list()):
            if isinstance(value, tuple | list) and len(value) == 2:
                low, high = float(value[0]), float(value[1])
                if math.isfinite(low) and math.isfinite(high):
                    lower[i] = means[i] - low
                    upper[i] = high - means[i]

    # A bootstrap CI need not be centred on the mean, but a negative offset
    # would make errorbar draw the whisker on the wrong side.
    return np.clip(np.vstack([lower, upper]), 0.0, None)


def draw_roi_panel(
    ax: Axes,
    roi_df: pd.DataFrame,
    model_order: list[str],
    main_outcome: str,
    model_colors: dict[str, str],
    model_labels: dict[str, str],
    *,
    show_xticklabels: bool = True,
    ylim_top: float | None = None,
) -> None:
    """Draw one ROI's benchmark bars onto an existing axis.

    Args:
        ax: Axis to draw on.
        roi_df: Rows for this ROI, one per model.
        model_order: Model names, in the order the bars should appear.
        main_outcome: The outcome column prefix being plotted.
        model_colors: Model name to bar colour.
        model_labels: Model name to display label.
        show_xticklabels: Whether to label the bars; off for inner grid panels.
        ylim_top: Explicit upper y limit, or None to fit the data.
    """
    rows = roi_df.set_index("model").loc[model_order].reset_index()
    means = rows[f"{main_outcome}_mean"].to_numpy(dtype=float)
    yerr = error_bar_bounds(rows, main_outcome)
    equivalent = rows["is_equivalent_to_best_on_score"].fillna(False).to_numpy()

    x = np.arange(len(model_order))
    ax.bar(
        x,
        means,
        color=[model_colors[model] for model in model_order],
        # Outline marks the models the equivalence test cannot separate from
        # the best one in this ROI.
        edgecolor=["k" if flag else "none" for flag in equivalent],
        linewidth=0.9,
    )
    ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="k", capsize=0)

    if "norm" in main_outcome:
        ax.axhline(y=1, color="black", linestyle="--", linewidth=1)

    top = ylim_top
    if top is None:
        top = float(np.nanmax(means + yerr[1])) * 1.05
        if "norm" in main_outcome:
            top = max(top, 1.05)
    ax.set_ylim(0, top)

    ax.set_xlim(-1, len(model_order))
    ax.set_xticks(x)
    if show_xticklabels:
        ax.set_xticklabels(
            [model_labels[model].replace("_", "-") for model in model_order],
            rotation=90,
        )
    else:
        ax.set_xticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    sns.despine(ax=ax)


@click.command()
@click.option(
    "--metric",
    type=str,
    default="rsa",
    show_default=True,
    help="Metric to use.",
)
@click.option(
    "--main-outcome",
    type=click.Choice(["score", "score_norm_upper", "score_norm_lower"]),
    default="score",
    show_default=True,
    help="Main outcome variable.",
)
@click.option(
    "--split",
    type=str,
    default="test",
    show_default=True,
    help="Data split.",
)
@click.option(
    "--plot-path",
    type=click.Path(exists=False, file_okay=False, path_type=Path),
    default=None,
    help="Output directory for plots (default: results/boldmoments/plots/YYYYMMDD).",
)
@click.option(
    "--taskonomy",
    type=bool,
    default=False,
    show_default=True,
    help="Whether to include taskonomy models.",
)
@click.option(
    "--dataset",
    type=click.Choice(["bold_moments", "nsd", "nsd_synthetic"]),
    default="bold_moments",
    show_default=True,
    help="Which dataset's canonical ROI and subject lists to use.",
)
@click.option(
    "--roi",
    type=str,
    default=None,
    help="Plot only this ROI. Default: every ROI of the dataset.",
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
    "--model-order-from",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    default=None,
    help=(
        "With --order global, rank the models on another aggregation instead "
        "of this one, so two datasets can be plotted on a shared x-axis. "
        "Either the $SCRATCH directory that aggregate_results wrote, or the "
        "subject_avg_df_equiv_score_norm_lower.pkl inside it."
    ),
)
@click.option(
    "--order",
    type=click.Choice(["global", "within-roi"]),
    default="global",
    show_default=True,
    help=(
        "Bar order. 'global' keeps figure_0's pooled ranking so panels line "
        "up; 'within-roi' sorts each panel by its own scores."
    ),
)
@click.option(
    "--ylim-top",
    type=float,
    default=None,
    help=(
        "Fix the upper y limit on every panel. Default fits each panel to its "
        "own data, which matters where a near-zero noise ceiling inflates the "
        "normalized scores."
    ),
)
@click.option(
    "--grid/--no-grid",
    default=True,
    show_default=True,
    help="Also write one overview figure with all ROIs as panels.",
)
def main(
    dataset: str,
    metric: str,
    main_outcome: str,
    split: str,
    plot_path: Path | None,
    taskonomy: bool,
    roi: str | None,
    exclude_rois: str,
    model_order_from: Path | None,
    order: str,
    ylim_top: float | None,
    grid: bool,
) -> None:
    """Write one benchmark bar chart per ROI, plus an optional overview grid."""
    rois = get_rois(dataset, exclude=parse_roi_names(exclude_rois))
    if roi is not None:
        if roi not in rois:
            raise click.BadParameter(
                f"ROI {roi!r} is not one of the {dataset} ROIs: {rois}",
                param_hint="--roi",
            )
        rois = [roi]

    scratch = Path(os.environ.get("SCRATCH", "/scratch"))
    project_root = Path(__file__).resolve().parent.parent.parent

    if plot_path is None:
        date_str = datetime.now().strftime("%Y%m%d")
        plot_path = project_root / "results" / "boldmoments" / "plots" / date_str
    plot_path.mkdir(parents=True, exist_ok=True)

    with open(project_root / "indiv_model_colors_dict.json") as handle:
        model_colors = json.load(handle)
    with open(project_root / "model_legend_mapping.json") as handle:
        model_labels = json.load(handle)

    order_scratch = order_pickle_dir(scratch, model_order_from)
    model_order = global_model_order(order_scratch, split, taskonomy)
    subject_avg_df = load_subject_avg_df(
        scratch, main_outcome, split, metric, taskonomy
    )
    if order_scratch != scratch:
        model_order = align_order_to_available(
            model_order,
            set(subject_avg_df["model"].unique()),
            str(order_scratch),
        )

    missing_colors = sorted(set(model_order) - set(model_colors))
    missing_labels = sorted(set(model_order) - set(model_labels))
    if missing_colors or missing_labels:
        raise click.ClickException(
            "Models are missing from the plotting metadata at the repo root. "
            f"indiv_model_colors_dict.json: {missing_colors}. "
            f"model_legend_mapping.json: {missing_labels}."
        )

    available = set(subject_avg_df["roi"].unique())
    missing_rois = [r for r in rois if r not in available]
    if missing_rois:
        raise click.ClickException(
            f"No {metric}/{main_outcome} rows for ROIs {missing_rois}. "
            f"The aggregated data covers: {sorted(available)}."
        )

    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 8,
        "lines.linewidth": 1,
        "lines.markersize": 5,
        "font.family": "serif",
    })
    panel_width = PANEL_WIDTH_TASKONOMY if taskonomy else PANEL_WIDTH

    for roi_name in rois:
        roi_df = subject_avg_df[subject_avg_df["roi"] == roi_name]
        bar_order = model_order
        if order == "within-roi":
            bar_order = (
                roi_df.sort_values(f"{main_outcome}_mean")["model"].tolist()
            )

        fig, ax = plt.subplots(figsize=(panel_width, PANEL_HEIGHT))
        draw_roi_panel(
            ax,
            roi_df,
            bar_order,
            main_outcome,
            model_colors,
            model_labels,
            ylim_top=ylim_top,
        )
        ax.set_title(roi_name)

        fname = f"figure_0_per_roi_{roi_name}_{metric}_{main_outcome}_{taskonomy}"
        save_this(
            path=plot_path, fname=fname, formats=["png"], dpi=300,
            transparent=False,
        )
        plt.close(fig)
        logger.info(f"Saved figure to {plot_path}/{fname}.*")

    if not grid or len(rois) < 2:
        return

    n_cols = min(GRID_COLUMNS, len(rois))
    n_rows = math.ceil(len(rois) / n_cols)

    # Only the lowest panel of each column is labelled, and with a ragged last
    # row that is not always the bottom row. Give just those rows the extra
    # height, so unlabelled rows are not padded out to match them.
    labelled_rows = {
        i // n_cols for i in range(len(rois)) if i + n_cols >= len(rois)
    }
    height_ratios = [
        PANEL_HEIGHT + (GRID_LABEL_BAND if row in labelled_rows else 0.0)
        for row in range(n_rows)
    ]
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_width * n_cols, sum(height_ratios)),
        gridspec_kw={"height_ratios": height_ratios},
        squeeze=False,
        # Not tight_layout: that derives a single hspace from the worst row,
        # so the label band under rows 4-5 would pad out every other row too.
        layout="constrained",
    )
    flat_axes = axes.ravel()

    for i, roi_name in enumerate(rois):
        roi_df = subject_avg_df[subject_avg_df["roi"] == roi_name]
        bar_order = model_order
        if order == "within-roi":
            bar_order = (
                roi_df.sort_values(f"{main_outcome}_mean")["model"].tolist()
            )
        draw_roi_panel(
            flat_axes[i],
            roi_df,
            bar_order,
            main_outcome,
            model_colors,
            model_labels,
            # Only the lowest panel in each column carries the model names.
            show_xticklabels=i + n_cols >= len(rois),
            ylim_top=ylim_top,
        )
        flat_axes[i].set_title(roi_name)

    for ax in flat_axes[len(rois):]:
        ax.set_axis_off()

    fname = f"figure_0_per_roi_grid_{metric}_{main_outcome}_{taskonomy}"
    save_this(
        path=plot_path, fname=fname, formats=["png"], dpi=300, transparent=False,
    )
    plt.close(fig)
    logger.info(f"Saved figure to {plot_path}/{fname}.*")


if __name__ == "__main__":
    main()
