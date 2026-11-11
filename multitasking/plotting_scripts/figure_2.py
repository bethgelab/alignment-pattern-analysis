#!/usr/bin/env python3
"""Generate figure 2 for boldmoments analysis.

Produces a matrix visualization showing which models are score-equivalent
and/or APS-equivalent to the best model across all ROIs, with model
property annotations.
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
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from multitasking.CONSTANTS import get_rois, parse_roi_names
from multitasking.plot_creation.plot_utils import save_this

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# The repo root, two levels up from multitasking/plotting_scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ATTRIBUTES_PATH = PROJECT_ROOT / "model_attributes.ods"


@click.command()
@click.option(
    "--metric",
    type=str,
    default="linear_predictivity",
    help="Metric to use (default: linear_predictivity)",
)
@click.option(
    "--main-outcome",
    type=click.Choice(["score", "score_norm_upper", "score_norm_lower"]),
    default="score_norm_lower",
    help="Main outcome variable (default: score_norm_lower)",
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
    help="Output directory for plots (default: results/boldmoments/plots/<date>)",
)
@click.option(
    "--dataset",
    type=click.Choice(["bold_moments", "nsd", "nsd_synthetic"]),
    default="bold_moments",
    show_default=True,
    help="Which dataset's canonical ROI list to use.",
)
@click.option(
    "--exclude-rois",
    type=str,
    default="",
    help=(
        "Comma-separated ROI names to drop entirely, e.g. 'V6,V6A'. Must "
        "match what aggregate_results was run with."
    ),
)
@click.option(
    "--model-attributes-path",
    default=str(DEFAULT_MODEL_ATTRIBUTES_PATH),
    show_default=True,
    help=("Spreadsheet of per-model attributes. Supplies the architecture, "
          "modality and supervision rows and the column ordering."),
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def main(metric, main_outcome, split, plot_path, dataset, exclude_rois,
         model_attributes_path):
    rois = get_rois(dataset, exclude=parse_roi_names(exclude_rois))
    date_str = datetime.now().strftime("%Y%m%d")

    scratch = Path(os.environ.get("SCRATCH", "/scratch"))
    project_root = Path(__file__).resolve().parent.parent.parent

    if plot_path is None:
        plot_path = project_root / "results" / "boldmoments" / "plots" / date_str
    plot_path.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    subject_avg_df = pickle.load(
        open(scratch / f"subject_avg_df_equiv_{main_outcome}.pkl", "rb")
    )

    subject_avg_df['model_provider'] = subject_avg_df['model'].str.split('/').str[0]
    subject_avg_df["predictor"] = (
        subject_avg_df["model"].astype(str) + "__" + subject_avg_df["layer"].astype(str)
    )

    # Build model attribute lookup from ODS file (architecture, modality, supervised, ...)
    model_attributes = pd.read_excel(model_attributes_path, engine="odf")
    model_attr_clean = model_attributes.dropna(subset=["model", "architecture"]).copy()
    model_attr_clean["model_prefix"] = model_attr_clean["model"].str.replace("*", "", regex=False)
    attr_lookup = {
        row["model_prefix"]: row.drop(["model", "model_prefix"]).to_dict()
        for _, row in model_attr_clean.iterrows()
    }

    def get_attributes(model_name):
        for prefix, attrs in attr_lookup.items():
            if isinstance(model_name, str) and model_name.startswith(prefix):
                return attrs
        return {}

    # Merge attributes into subject_avg_df (needed for sorting by architecture/modality/supervised)
    attr_expanded = subject_avg_df["model"].apply(get_attributes).apply(pd.Series)
    subject_avg_df = pd.concat(
        [subject_avg_df.reset_index(drop=True), attr_expanded.reset_index(drop=True)], axis=1
    )

    # Load model legend mapping. The repo-root file is keyed by model name;
    # the version this script used to read lived in another project and was
    # zipped positionally against subject_avg_df["model"].unique(), so any
    # change to the model set silently relabelled every column.
    with open(PROJECT_ROOT / "model_legend_mapping.json") as f:
        model_legend_mapping = json.load(f)

    # Filter data
    panel_1_df = subject_avg_df[
        (subject_avg_df['split'] == split)
        & (subject_avg_df['metric'] == metric)
        & (subject_avg_df['is_best_layer'])
        & (subject_avg_df['model_provider'] != 'taskonomy')
    ]

    # Build equivalence matrices (ROIs x models)
    score_equiv_matrix = (
        panel_1_df
        .pivot_table(
            index="roi",
            columns="model",
            values="is_equivalent_to_best_on_score",
            aggfunc="any",
            fill_value=False,
        )
        .astype(bool)
    )

    aps_equiv_matrix = (
        panel_1_df
        .pivot_table(
            index="roi",
            columns="model",
            values="is_equivalent_to_best_on_aps",
            aggfunc="any",
            fill_value=False,
        )
        .astype(bool)
    )

    # Reorder columns by model attributes, rows by canonical ROI order
    models_ordered = (
        subject_avg_df.groupby('model').first()
        .sort_values(by=['architecture', 'modality', 'supervised'])
        .index.to_list()
    )
    score_equiv_matrix = score_equiv_matrix.reindex(columns=models_ordered, fill_value=False)
    score_equiv_matrix = score_equiv_matrix.reindex(rois)
    aps_equiv_matrix = aps_equiv_matrix.reindex(columns=models_ordered, fill_value=False)
    aps_equiv_matrix = aps_equiv_matrix.reindex(rois)

    # Build categorical matrix: 0=not equivalent, 1=score only, 2=score+aps
    score = score_equiv_matrix.values
    aps = aps_equiv_matrix.values

    combined = np.zeros_like(score, dtype=int)
    combined[np.logical_and(score, ~aps)] = 1
    combined[np.logical_and(score, aps)] = 2

    # Keep only models with at least one equivalence entry
    models_to_keep_mask = (combined > 0).any(axis=0)
    combined = combined[:, models_to_keep_mask]
    score_equiv_matrix = score_equiv_matrix.loc[:, models_to_keep_mask]
    aps_equiv_matrix = aps_equiv_matrix.loc[:, models_to_keep_mask]
    models_to_keep = score_equiv_matrix.columns
    logger.info(f"Models to keep: {list(models_to_keep)}")

    # Build model property matrix (CNN, image, supervised)
    n_props = 3
    model_props = np.ones((len(models_to_keep), n_props))
    for i, model in enumerate(models_to_keep):
        attrs = get_attributes(model)
        model_props[i, 0] = attrs.get('architecture') == "CNN"
        model_props[i, 1] = attrs.get('modality') == "image"
        model_props[i, 2] = attrs.get('supervised') == 'yes'

    # Compute layout dimensions before figure creation
    nrows, ncols = combined.shape

    # Plot settings
    plt.rcParams.update(
        {
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
        }
    )

    cell_size = 0.15  # inches per cell
    fig = plt.figure(figsize=(ncols * cell_size, (nrows + n_props) * cell_size))
    gs = fig.add_gridspec(2, 1, height_ratios=[nrows, n_props])

    ax_main = fig.add_subplot(gs[0, 0])
    ax_props = fig.add_subplot(gs[1, 0], sharex=ax_main)

    # Main heatmap: ROIs (rows) x models (columns)
    cmap_main = ListedColormap(["white", "lightgray", "black"])

    mat = combined

    ax_main.pcolormesh(
        np.arange(ncols + 1), np.arange(nrows + 1), mat,
        cmap=cmap_main, edgecolors="white", linewidth=1,
    )
    ax_main.set_xlim(0, ncols)
    ax_main.set_ylim(nrows, 0)
    ax_main.set_aspect("equal")
    ax_main.xaxis.set_visible(False)
    ax_main.set_yticks(np.arange(0.5, len(rois) + 0.5), rois)
    ax_main.yaxis.set_tick_params(pad=1)

    # Model properties panel
    n_models, n_props = model_props.shape
    cmap_props = ListedColormap(["lightblue", "darkblue"])
    ax_props.pcolormesh(
        np.arange(n_models + 1), np.arange(n_props + 1),
        model_props.astype(int).T,
        cmap=cmap_props, edgecolors="white", linewidth=1,
    )
    ax_props.set_aspect("equal")
    ax_props.set_yticks(np.arange(0.5, n_props + 0.5), ['CNN', 'Image', 'Supervised'])
    ax_props.set_xticks(
        np.arange(0.5, len(models_to_keep) + 0.5),
        [model_legend_mapping.get(m, m).replace("_", "-") for m in models_to_keep],
        rotation=90,
    )
    ax_props.xaxis.set_tick_params(pad=1)
    ax_props.yaxis.set_tick_params(pad=1)

    # Legend
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='white', label='no'),
        Patch(facecolor='darkblue', edgecolor='white', label='yes'),
        Patch(facecolor='white', edgecolor='black', linestyle='--', label='not equivalent'),
        Patch(facecolor='lightgray', edgecolor='white', label='equivalent on score'),
        Patch(facecolor='black', edgecolor='white', label='equivalent on score and aps'),
    ]
    ax_props.legend(
        handles=legend_elements,
        loc='center left', bbox_to_anchor=(1, 0.5),
        handlelength=1, handleheight=1, frameon=False,
    )

    save_this(path=plot_path, fname=f"figure_2_{main_outcome}_{metric}",
               formats=["png"], transparent=False)
    logger.info(f"Saved figure to {plot_path}/figure_2_{main_outcome}_{metric}.*")


if __name__ == "__main__":
    main()
