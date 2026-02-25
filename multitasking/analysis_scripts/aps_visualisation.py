#!/usr/bin/env python3
"""Script to visualize Alignment Pattern Similarity (APS) scores vs model performance scores.

This script generates scatter plots comparing APS scores with model performance scores,
grouped by model families and ROI groups.
"""

import argparse
import logging
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from multitasking.CONSTANTS import ROIS as rois
from multitasking.plot_creation.dataframe_utils import select_top_layer
from multitasking.plot_creation.plot_utils import save_this

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(scratch_dir: Path, metric: str):
    """Load all required data files."""
    logger.info(f"Loading data for metric: {metric}")

    # Load dataframes
    with open(scratch_dir / "equivalent_models_df.pkl", "rb") as f:
        equivalent_models_df = pickle.load(f)

    with open(scratch_dir / "bootstrap_ci.pkl", "rb") as f:
        bootstrap_ci = pickle.load(f)

    with open(scratch_dir / "full_df.pkl", "rb") as f:
        full_df = pickle.load(f)

    with open(scratch_dir / "df.pkl", "rb") as f:
        df = pickle.load(f)

    with open(scratch_dir / "model_by_roi_matrix.pkl", "rb") as f:
        model_by_roi_matrix = pickle.load(f)

    # Load model-brain APS data
    aps_path = scratch_dir / "iclr_analyses_aps_trained_models"
    with open(aps_path / f"all_model_layers_model_brain_aps_{metric}.pkl", "rb") as f:
        model_brain_alignment_patterns = pickle.load(f)

    # Load color dictionaries
    with open(scratch_dir / "model_colors_dict.pkl", "rb") as f:
        model_colors_dict = pickle.load(f)

    with open(scratch_dir / "roi_colors_dict.pkl", "rb") as f:
        roi_color_dict = pickle.load(f)

    return {
        "equivalent_models_df": equivalent_models_df,
        "bootstrap_ci": bootstrap_ci,
        "full_df": full_df,
        "df": df,
        "model_by_roi_matrix": model_by_roi_matrix,
        "model_brain_alignment_patterns": model_brain_alignment_patterns,
        "model_colors_dict": model_colors_dict,
        "roi_color_dict": roi_color_dict,
    }


def create_aps_dataframe(model_brain_alignment_patterns,
                         metric: str, subjects: list[str] | None = None):
    """Create a dataframe from APS alignment pattern similarity data."""
    logger.info("Creating APS dataframe")
    rows = []
    if subjects is None:
        for roi, model_dict in model_brain_alignment_patterns.alignment_pattern_similarity.items():
            for model_layer_key, aps_scores in model_dict.items():
                model, layer = model_layer_key.split("__", 1)
                rows.append({
                    "roi": roi,
                    "model": model,
                    "layer": layer,
                    "metric": metric,
                    "aps_score_mean": np.array(aps_scores).mean(),
                    "aps_score_std": np.array(aps_scores).std(),
                })
        return pd.DataFrame(rows)
    else:
        for roi, model_dict in model_brain_alignment_patterns.alignment_pattern_similarity.items():
            for model_layer_key, aps_scores in model_dict.items():
                model, layer = model_layer_key.split("__", 1)
                for i_subject, subject in enumerate(subjects):
                    rows.append({
                        "roi": roi,
                        "model": model,
                        "layer": layer,
                        "metric": metric,
                        "subject": subject,
                        "aps_score": aps_scores[i_subject],
                    })
        return pd.DataFrame(rows)


def prepare_joint_dataframe(full_df, aps_df, metric: str):
    """Prepare the joint dataframe combining scores and APS data."""
    logger.info("Preparing joint dataframe")

    # Aggregate subject data
    subject_avg_df = full_df.groupby(
        ["model", "layer", "roi", "split", "metric", "stream"], as_index=False
    ).agg(
        count=("score", "count"),
        score_mean=("score", "mean"),
        score_std=("score", "std"),
    )

    # Get best layers
    _, roi_layer_map = select_top_layer(subject_avg_df, split="train")
    roi_layer_map.insert(roi_layer_map.shape[1], 'is_best_layer', True)

    # Merge with APS data
    joint_df = subject_avg_df.query(f"metric == '{metric}' and split == 'test'").merge(
        aps_df, on=["model", "layer", "roi", "metric"], how="left"
    )

    # Merge with best layer info
    joint_df = joint_df.merge(roi_layer_map, on=["roi", "layer", "metric", "model"], how="outer")
    boolean_layer = joint_df['is_best_layer'].apply(lambda x: False if np.isnan(x) else x)
    joint_df.drop("is_best_layer", axis=1, inplace=True)
    joint_df.insert(joint_df.shape[1], 'is_best_layer', boolean_layer)

    # Add model family
    joint_df["model_family"] = joint_df["model"].apply(lambda x: x.split("/")[0])

    # Add ordered ROIs
    joint_df['ordered_rois'] = pd.Categorical(joint_df['roi'], categories=rois, ordered=True)

    return joint_df


def create_plots(joint_df, roi_color_dict, plot_path: Path, metric: str):
    """Create scatter plots comparing scores vs APS scores."""
    logger.info("Creating plots")

    # Define ROI groups
    roi_groups = [rois[:3], rois[3:7], rois[7:]]

    # Get model families and their models
    family_to_models = (
        joint_df[['model_family', 'model']]
        .drop_duplicates()
        .groupby('model_family')['model']
        .apply(list)
        .to_dict()
    )

    # Create plots for each model family
    for model_family, models in family_to_models.items():
        logger.info(f"Creating plots for model family: {model_family}")

        # Filter out problematic models
        models = [m for m in models if m != "taskonomy/denoising"]

        if not models:
            continue

        fig_height = max(2, (15 // 6) * len(models))
        fig, axes = plt.subplots(
            nrows=len(models),
            ncols=len(roi_groups),
            figsize=(10, fig_height),
            sharex=True,
            sharey=True
        )

        if axes.ndim == 1:
            axes = axes.reshape(1, -1)

        legend = True
        for i, model in enumerate(models):
            for j, roi_group in enumerate(roi_groups):
                ax = axes[i, j]

                # Plot non-best layers
                sns.scatterplot(
                    data=joint_df.query(
                        f"metric == '{metric}' and model_family == @model_family "
                        f"and not is_best_layer and roi in @roi_group and model == @model"
                    ),
                    y="aps_score_mean",
                    x="score_mean",
                    hue="ordered_rois",
                    style="is_best_layer",
                    palette=roi_color_dict,
                    markers=["o"],
                    ax=ax,
                    legend=legend,
                )

                # Plot best layers
                sns.scatterplot(
                    data=joint_df.query(
                        f"metric == '{metric}' and model_family == @model_family "
                        f"and is_best_layer and roi in @roi_group and model == @model"
                    ),
                    y="aps_score_mean",
                    x="score_mean",
                    hue="ordered_rois",
                    palette=roi_color_dict,
                    marker="X",
                    ax=ax,
                    legend=False,
                )

                # Customize legend for first plot
                if legend:
                    ax.legend_.remove()
                    handles = ax.get_legend_handles_labels()
                    current_handles = [h for h in handles[0] if h.get_label() in roi_group]
                    current_labels = [label for label in handles[1] if label in roi_group]
                    ax.legend(
                        current_handles,
                        current_labels,
                        borderaxespad=0.,
                        frameon=False,
                        # bbox_to_anchor=(0.5, 1.02),
                        ncols=max(1, int(np.ceil(len(roi_group) // 4)))
                    )

            legend = False
            axes[i, 1].set_title(model)

        sns.despine()
        plt.tight_layout()

        # Save plot
        save_this(
            plot_path,
            fname=f"score_vs_aps_model_family_{model_family}_{metric}",
            formats=['png'],
            transparent=False,
        )
        plt.close()

        logger.info(f"Saved plot for {model_family}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Visualize APS scores vs model performance scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="rsa",
        choices=["rsa", "linear_predictivity"],
        help="Metric to use for visualization (default: rsa)",
    )
    parser.add_argument(
        "--scratch-dir",
        type=str,
        default=None,
        help="Path to scratch directory containing data files (default: $SCRATCH or /scratch)",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default=None,
        help="Path to save plots (default: results/boldmoments/plots/score_vs_aps)",
    )

    args = parser.parse_args()

    # Set up paths
    if args.scratch_dir is None:
        scratch_dir = Path(os.environ.get("SCRATCH", "/scratch"))
    else:
        scratch_dir = Path(args.scratch_dir)

    if args.plot_path is None:
        plot_path = Path(
            "/mnt/lustre/work/bethge/bkr578/projects/multitasking/results/boldmoments/plots/20260119/score_vs_aps"
        )
    else:
        plot_path = Path(args.plot_path)

    plot_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Using scratch directory: {scratch_dir}")
    logger.info(f"Saving plots to: {plot_path}")
    logger.info(f"Using metric: {args.metric}")

    # Load data
    data = load_data(scratch_dir, args.metric)

    # Create APS dataframe
    aps_df = create_aps_dataframe(data["model_brain_alignment_patterns"], args.metric)


    # Prepare joint dataframe
    joint_df = prepare_joint_dataframe(data["full_df"], aps_df, args.metric)

    # save aps_df
    pickle.dump(joint_df, open(scratch_dir / f"joint_df_{args.metric}.pkl", "wb"))

    # Create plots
    create_plots(joint_df, data["roi_color_dict"], plot_path, args.metric)

    logger.info("Done!")


if __name__ == "__main__":
    main()
