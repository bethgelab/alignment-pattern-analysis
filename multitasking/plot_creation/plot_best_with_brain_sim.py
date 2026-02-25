"""Plot top model scores with brain-brain similarities, color-coded by subject.

This script creates visualizations showing:
- Top N model scores for each ROI (as box plots + subject-colored scatter points)
- Top M ROI-to-ROI similarities (as subject-colored diamond markers)
- Noise ceiling for each ROI

"""

import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from multitasking.plot_creation.dataframe_utils import (
    add_stream_hierarchy_info,
    select_top_layer,
)
from multitasking.plot_creation.plot_utils import save_this
from multitasking.utils.scoresheet_loading import load_scoresheets_from_parent_dir

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.WARNING)

# Suppress verbose logging from scoresheet_loading module
logging.getLogger("multitasking.utils.scoresheet_loading").setLevel(logging.WARNING)

def _filter_by_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Filter a DataFrame by metric, ignoring casing and underscores.
    
    Parameters
    ----------
    df : pd.DataFrame, DataFrame to filter (on column "metric")
    metric : str, metric name to filter by.

    Returns:
    -------
    pd.DataFrame, filtered DataFrame.
    """
    return df[df["metric"].str.lower().str.replace("_", "") == metric.lower().replace("_", "")]


def _check_intersubject_results(df: pd.DataFrame) -> None:
    """Validate intersubject results DataFrame structure."""
    rois = df["roi"].unique()
    rois_2 = df["layer"].unique()
    if set(rois) != set(rois_2):
        LOGGER.warning(
            f"ROI mismatch: {set(rois) - set(rois_2)} in roi but not in layer, "
            f"and {set(rois_2) - set(rois)} in layer but not in roi"
        )
    
    models = df["model"].unique()
    if len(models) == 1 and models[0] != "all-other-subjects":
        LOGGER.warning(
            f"Expected 'all-other-subjects' as model in intersubject results, "
            f"found {models[0]}"
        )


def load_model_results(
    model_results_dir: Path,
    # filename: str,
    split: str,
    metric: str
) -> pd.DataFrame:
    """Load and process model benchmark results.

    Returns DataFrame with subject-level scores (not aggregated).
    """
    LOGGER.info(f"Loading model results from {model_results_dir}")

    # Load scoresheets
    df = load_scoresheets_from_parent_dir(
        model_results_dir,
        filename=None,
        filename_metric=metric,
    )

    if len(df) == 0:
        raise ValueError(f"No model results found in {model_results_dir}")

    LOGGER.warning(f"Loaded {len(df)} model result rows")

    # Debug: Check which subjects have which models
    LOGGER.warning(f"Unique subjects: {sorted(df['subject'].unique())}")
    LOGGER.warning(f"Unique models: {len(df['model'].unique())} models")

    # Check if vggt/VGGT-1B exists and for which subjects
    vggt_data = df[df['model'] == 'vggt/VGGT-1B']
    if len(vggt_data) > 0:
        vggt_subjects = sorted(vggt_data['subject'].unique())
        LOGGER.warning(f"Model 'vggt/VGGT-1B' found for subjects: {vggt_subjects}")
    else:
        LOGGER.warning("Model 'vggt/VGGT-1B' not found in any data!")

    # Add stream hierarchy info before filtering
    df = add_stream_hierarchy_info(df)

    # Average across subjects first (like summary_plots.py)
    subject_avg_df = df.groupby(
        ["model", "layer", "roi", "split", "metric", "stream", "order"],
        as_index=False
    ).agg(
        count=("score", "count"),
        score_mean=("score", "mean"),
        score_std=("score", "std"),
    )

    # Select top layer per model-ROI combination using train split on averaged data
    df_top_layer, _ = select_top_layer(subject_avg_df, split="train")

    # Now merge back with original subject-level data to get individual scores
    # Keep only the model-layer-roi-split combinations that were selected as best
    # Note: df_top_layer contains both train and test splits for the selected layers

    # Check for and remove duplicates in original data
    # Duplicates can occur if multiple scoresheet files contain the same results
    dup_check = df.duplicated(subset=["model", "layer", "roi", "metric", "split", "subject"])
    if dup_check.any():
        n_dups = dup_check.sum()
        LOGGER.warning(f"Found {n_dups} rows with duplicate metadata (model/layer/roi/split/subject)")

        # Check if these duplicates also have identical scores
        # Group by metadata and check score variance
        duplicate_groups = df[df.duplicated(subset=["model", "layer", "roi", "metric", "split", "subject"], keep=False)]
        if len(duplicate_groups) > 0:
            grouped = duplicate_groups.groupby(["model", "layer", "roi", "metric", "split", "subject"])
            score_variance = grouped["score"].std()
            max_variance = score_variance.max()
            score_min = grouped["score"].min()
            score_max = grouped["score"].max()

            if pd.notna(max_variance) and max_variance > 1e-6:
                LOGGER.warning(
                    f"WARNING: Some duplicates have different scores! Max score std: {max_variance:.6f}"
                )
                # Show examples of duplicates with different scores
                high_var_groups = score_variance[score_variance > 1e-6].head(3)
                for idx, var in high_var_groups.items():
                    example = df[(df["model"] == idx[0]) &
                                (df["layer"] == idx[1]) &
                                (df["roi"] == idx[2]) &
                                (df["metric"] == idx[3]) &
                                (df["split"] == idx[4]) &
                                (df["subject"] == idx[5])]
                    LOGGER.warning(
                        f"  Example: {idx[0]} / {idx[1]} / {idx[2]} / {idx[5]} "
                        f"has scores: {example['score'].tolist()}"
                    )
                raise AssertionError("See above!")
            else:
                LOGGER.info("All duplicates have identical scores (within tolerance)")
            if pd.notna(max_variance) and not np.allclose(score_min, score_max, atol=1e-6, equal_nan=True):
                raise AssertionError("Rows with duplicate metadata have different scores!"
                                f"Min score: {score_min:.6f}, Max score: {score_max:.6f}"
                                f"Difference: {score_max - score_min:.6f}")


        # Remove duplicates, keeping first occurrence
        df = df.drop_duplicates(
            subset=["model", "layer", "roi", "metric", "split", "subject"],
            keep="first", 
        )
        LOGGER.warning(f"After deduplication: {len(df)} rows remaining")

        # Check vggt after deduplication
        vggt_data_after = df[df['model'] == 'vggt/VGGT-1B']
        if len(vggt_data_after) > 0:
            vggt_subjects_after = sorted(vggt_data_after['subject'].unique())
            LOGGER.warning(f"After dedup, 'vggt/VGGT-1B' found for subjects: {vggt_subjects_after}")

    df_processed = df.merge(
        df_top_layer[["model", "layer", "roi", "metric", "split"]],
        on=["model", "layer", "roi", "metric", "split"],
        how="inner"
    )

    # Check vggt after merge with top layers
    vggt_processed = df_processed[df_processed['model'] == 'vggt/VGGT-1B']
    if len(vggt_processed) > 0:
        vggt_subjects_processed = sorted(vggt_processed['subject'].unique())
        vggt_rois = sorted(vggt_processed['roi'].unique())
        LOGGER.warning(f"After layer selection, 'vggt/VGGT-1B' has subjects: {vggt_subjects_processed}")
        LOGGER.warning(f"  ROIs where vggt is a top layer: {vggt_rois}")
    else:
        LOGGER.warning("After layer selection, 'vggt/VGGT-1B' was filtered out completely!")

    # Filter to specified split and metric
    df_processed = df_processed[df_processed["split"] == split]
    # df_processed = df_processed[df_processed["metric"].str.lower().str.replace("_", "") \
    #                                             == metric.lower().replace("_", "")]
    df_processed = _filter_by_metric(df_processed, metric)

    if len(df_processed) == 0:
        raise ValueError(
            f"No results found for split='{split}' and metric='{metric}'."
            f"Available metrics in the original loaded scoresheets: {df['metric'].unique()}"
        )
    
    LOGGER.info(
        f"Processed to {len(df_processed)} rows across "
        f"{df_processed['subject'].nunique()} subjects"
    )
    
    return df_processed


def load_intersubject_similarities(
    intersubject_results_dir: Path,
    # filename: str,
    split: str,
    metric: str
) -> pd.DataFrame:
    """Load ROI-to-ROI similarity data from intersubject consistency results.

    Returns DataFrame with subject-level scores (not aggregated).
    """
    LOGGER.info(f"Loading intersubject results from {intersubject_results_dir}")

    # Load scoresheets
    df = load_scoresheets_from_parent_dir(
        intersubject_results_dir,
        filename=None,
        filename_metric=metric,
    )

    if df is None or len(df) == 0:
        raise ValueError(
            f"No intersubject results found in {intersubject_results_dir}"
        )
    
    LOGGER.info(f"Loaded {len(df)} intersubject result rows")
    
    # Validate structure
    _check_intersubject_results(df)
    
    # Filter to specified split and metric
    df = df[df["split"] == split]
    df = _filter_by_metric(df, metric)
    # df = df[df["metric"].str.lower().str.replace("_", "") == metric.lower().replace("_", "")]
    
    if len(df) == 0:
        raise ValueError(
            f"No intersubject results found for split='{split}' and metric='{metric}'."
            f"Available metrics in the original loaded scoresheets: {df['metric'].unique()}"
        )
    
    LOGGER.info(
        f"Filtered to {len(df)} rows across {df['subject'].nunique()} subjects"
    )
    
    return df


def get_top_n_models_per_roi(
    model_df: pd.DataFrame,
    roi: str,
    n: int = 10
) -> pd.DataFrame:
    """Get top N models for a specific ROI based on subject-averaged scores.

    Args:
        model_df: DataFrame with model results (subject-level)
        roi: Target ROI name
        n: Number of top models to return (-1 for all)

    Returns:
        DataFrame with columns: model, score, subject
    """
    # Filter to this ROI
    roi_df = model_df[model_df["roi"] == roi].copy()

    if len(roi_df) == 0:
        LOGGER.warning(f"No model results found for ROI {roi}")
        return pd.DataFrame(columns=["model", "score", "subject"])

    # Compute mean score per model across subjects (like summary_plots.py)
    model_means = roi_df.groupby("model")["score"].mean().sort_values(ascending=False)

    # Select top N models based on averaged scores
    if n == -1:
        top_models = model_means.index.tolist()
    else:
        top_models = model_means.head(n).index.tolist()

    # Get all subject-level scores for these top models
    top_model_data = roi_df[roi_df["model"].isin(top_models)][
        ["model", "score", "subject"]
    ].copy()

    LOGGER.debug(
        f"ROI {roi}: Selected {len(top_models)} models with "
        f"{len(top_model_data)} total data points"
    )

    return top_model_data


def get_top_n_roi_similarities(
    intersubject_df: pd.DataFrame,
    target_roi: str,
    n: int = 4,
    include_self: bool = True
) -> pd.DataFrame:
    """Get top N ROI-to-ROI similarities for a target ROI.

    Args:
        intersubject_df: DataFrame with intersubject results (subject-level)
        target_roi: Target ROI name
        n: Number of top similarities to return
        include_self: If True, include self-similarity (noise ceiling) in the count

    Returns:
        DataFrame with columns: layer (source ROI), score, subject
    """
    # Filter to this target ROI
    roi_df = intersubject_df[intersubject_df["roi"] == target_roi].copy()

    if not include_self:
        # Exclude self-similarity (noise ceiling)
        roi_df = roi_df[roi_df["layer"] != roi_df["roi"]]

    if len(roi_df) == 0:
        LOGGER.warning(f"No ROI similarities found for ROI {target_roi}")
        return pd.DataFrame(columns=["layer", "score", "subject"])

    # Compute mean score per source ROI across subjects
    roi_means = roi_df.groupby("layer")["score"].mean().sort_values(ascending=False)

    # Select top N source ROIs
    top_source_rois = roi_means.head(n).index.tolist()

    # Get all subject-level scores for these top source ROIs
    top_sim_data = roi_df[roi_df["layer"].isin(top_source_rois)][
        ["layer", "score", "subject"]
    ].copy()

    LOGGER.debug(
        f"ROI {target_roi}: Selected {len(top_source_rois)} source ROIs with "
        f"{len(top_sim_data)} total data points"
    )

    return top_sim_data


def get_noise_ceiling(
    intersubject_df: pd.DataFrame,
    roi: str
) -> pd.DataFrame:
    """Get noise ceiling for a specific ROI (self-similarity).
    
    Args:
        intersubject_df: DataFrame with intersubject results
        roi: Target ROI name
    
    Returns:
        DataFrame with columns: score, subject
    """
    # Filter to self-similarity (noise ceiling)
    nc_df = intersubject_df[
        (intersubject_df["roi"] == roi) & (intersubject_df["layer"] == roi)
    ][["score", "subject"]].copy()
    
    if len(nc_df) == 0:
        LOGGER.warning(f"No noise ceiling found for ROI {roi}")
    
    return nc_df


def create_roi_comparison_plot(
    model_df: pd.DataFrame,
    intersubject_df: pd.DataFrame,
    rois: list[str],
    metric: str,
    split: str,
    plot_path: Path,
    formats: tuple[str, ...],
    plot_fname: str,
    n_top_models: int = 10,
    n_top_similarities: int = 4,
    per_roi_plots: bool = False,
    plot_difference: bool = False,
):
    """Create visualization(s) showing model scores and brain-brain similarities.
    
    Args:
        model_df: Model results DataFrame (subject-level)
        intersubject_df: Intersubject results DataFrame (subject-level)
        rois: List of ROI names to plot
        metric: Metric name
        split: Split name
        plot_path: Output directory
        formats: Output formats
        plot_fname: Base filename
        n_top_models: Number of top models to show
        n_top_similarities: Number of top ROI similarities to show
        per_roi_plots: If True, create separate plot per ROI
        plot_difference: If True, plot the difference between same-same 
                        similarity and same-other similarity, instead of 
                        raw values.
    """
    # Setup subject color palette
    subjects = sorted(model_df["subject"].unique())
    n_subjects = len(subjects)
    subject_colors = sns.color_palette("tab10", n_subjects)
    subject_color_map = dict(zip(subjects, subject_colors))

    # Use a single consistent color for all best subject markers
    best_subject_marker_color = 'darkgreen'

    LOGGER.info(f"Creating plots with {n_subjects} subjects: {subjects}")
    
    if per_roi_plots:
        # Create one plot per ROI
        for roi in rois:
            _plot_single_roi(
                roi, model_df, intersubject_df, subjects, subject_color_map,
                metric, split, plot_path, formats, plot_fname,
                n_top_models, n_top_similarities, best_subject_marker_color,
                plot_difference,
            )
    else:
        # Create combined plot with all ROIs
        _plot_all_rois(
            rois, model_df, intersubject_df, subjects, subject_color_map,
            metric, split, plot_path, formats, plot_fname,
            n_top_models, n_top_similarities, best_subject_marker_color,
            plot_difference,
        )

def subtract_per_subject(df: pd.DataFrame, 
                        to_subtract_df: pd.DataFrame, 
                        on="subject",
                        subract_col: str = "score") -> pd.DataFrame:
    """Subtract the values of to_subtract_df from df, matched on column `on`.
    
        dataframe to_subtract_df must have N columns where N is the number of unique 
        values in the column `on` of df.
    """
    if len(to_subtract_df) != len(df[on].unique()):
        raise ValueError(f"to_subtract_df must have the same number of unique values in column {on} as df")
    to_subtract_df = to_subtract_df.copy().set_index(on)
    df = df.copy()
    df["_to_subtract"] = df.apply(lambda row: to_subtract_df.loc[row[on], subract_col], axis=1)
    df[subract_col] = df[subract_col] - df["_to_subtract"]
    df = df.drop(columns=["_to_subtract"])
    return df



def _plot_all_rois(
    rois: list[str],
    model_df: pd.DataFrame,
    intersubject_df: pd.DataFrame,
    subjects: list[str],
    subject_color_map: dict,
    metric: str,
    split: str,
    plot_path: Path,
    formats: tuple[str, ...],
    plot_fname: str,
    n_top_models: int,
    n_top_similarities: int,
    best_subject_marker_color: str = 'darkgreen',
    plot_difference: bool = False,
):
    """Create combined plot with all ROIs side-by-side."""
    n_rois = len(rois)
    # Calculate width: need space for model boxplot + ROI boxplots per ROI
    fig_width = max(16, (n_rois + n_top_similarities) * 0.75)
    fig, ax = plt.subplots(figsize=(fig_width, 10))

    all_positions = []
    all_labels = []
    current_pos = 0  # Track the current position as we plot

    for roi_idx, roi in enumerate(rois):
        LOGGER.debug(f"Plotting ROI {roi} at position {roi_idx}")

        # Add spacing between ROIs (except for the first one)
        if roi_idx > 0:
            current_pos += 1  # Add gap between ROIs

        # 0. Get same-same similarity for this ROI if needed
        if plot_difference:
            same_same_similarity = get_noise_ceiling(intersubject_df, roi)

        # A. Get top N model scores
        top_model_data = get_top_n_models_per_roi(model_df, roi, n=n_top_models)
        if plot_difference:
            top_model_data = subtract_per_subject(top_model_data, same_same_similarity)

        best_subject = None
        top_models_list = []
        n_models_to_plot = 0
        
        if len(top_model_data) > 0:
            # Get the list of top models for this ROI, ordered by mean score
            model_means = top_model_data.groupby("model")["score"].mean().sort_values(ascending=True)
            top_models_list = model_means.index.tolist()
            if len(top_models_list) != n_top_models:
                LOGGER.warning(f"ROI {roi}: Only {len(top_models_list)} top models found, expected {n_top_models}")
                n_models_to_plot = len(top_models_list)
            else:
                n_models_to_plot = n_top_models

            # Find the subject with the highest mean model score for this ROI
            subject_means = top_model_data.groupby("subject")["score"].mean()
            best_subject = subject_means.idxmax()

            # Check if best subject has all expected models (do this once before the loop)
            best_subject_top_models = top_model_data[top_model_data["subject"] == best_subject]["model"].unique()
            if len(best_subject_top_models) != len(top_models_list):
                missing_models = set(top_models_list) - set(best_subject_top_models)
                LOGGER.warning(
                    f"ROI {roi}: Best subject {best_subject} has only {len(best_subject_top_models)}/{len(top_models_list)} "
                    f"models. Missing models: {sorted(missing_models)}"
                )

        # Set base position for this ROI
        base_pos = current_pos

        # Plot each model as a separate boxplot
        if len(top_model_data) > 0:
            for model_idx, model in enumerate(top_models_list):
                x_pos = base_pos + model_idx
                model_data = top_model_data[top_model_data["model"] == model]
                
                # Create boxplot for this model's scores across subjects
                bp = ax.boxplot(
                    [model_data["score"].values],
                    positions=[x_pos],
                    widths=0.8,
                    patch_artist=True,
                    showfliers=False,
                    whis=[0, 100],  # min/max whiskers
                    boxprops=dict(facecolor='steelblue', alpha=0.6, edgecolor='darkblue'),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(color='darkblue', linewidth=1.5),
                    capprops=dict(color='darkblue', linewidth=1.5),
                )

                # Add scatter point for the best subject
                best_subject_model_data = model_data[model_data["subject"] == best_subject]
                if len(best_subject_model_data) > 0:
                    best_subject_score = best_subject_model_data["score"].values[0]
                    ax.scatter(
                        x_pos,
                        best_subject_score,
                        color=best_subject_marker_color,
                        edgecolor='black',
                        linewidth=0.5,
                        alpha=0.9,
                        s=50,
                        zorder=15,
                        label='Best subject' if roi_idx == 0 and model_idx == 0 else None
                    )
                else:
                    LOGGER.warning(
                        f"ROI {roi}: Best subject {best_subject} missing data for model {model}"
                    )

                # Shorten model name for label
                if model.startswith('clip/'):
                    short_name = model
                elif '/' in model:
                    parts = model.split('/')
                    short_name = '/'.join(parts[1:])
                else:
                    short_name = model
                
                all_positions.append(x_pos)
                all_labels.append(f'{roi}\n{short_name}' if model_idx == 0 else short_name)

        # B. Plot top M ROI-to-ROI similarities (including self if among top)
        top_sim_data = get_top_n_roi_similarities(
            intersubject_df, roi, n=n_top_similarities, include_self=True
        )
        if plot_difference:
            top_sim_data = subtract_per_subject(top_sim_data, same_same_similarity)

        n_sim_boxes_plotted = 0
        if len(top_sim_data) > 0:
            # Get unique source ROIs ordered by mean score
            source_rois = (
                top_sim_data.groupby("layer")["score"]
                .mean()
                .sort_values(ascending=False)
                .index.tolist()
            )
            n_sim_boxes_plotted = len(source_rois)

            # Plot each source ROI as a separate boxplot
            for idx, src_roi in enumerate(source_rois):
                x_pos = base_pos + n_models_to_plot + idx
                src_roi_data = top_sim_data[top_sim_data["layer"] == src_roi]

                # Determine color: use different color if it's the noise ceiling
                is_self = src_roi == roi
                facecolor = 'gold' if is_self else 'lightcoral'
                edgecolor = 'darkgoldenrod' if is_self else 'darkred'

                # Create boxplot for this ROI-to-ROI similarity
                bp = ax.boxplot(
                    [src_roi_data["score"].values],
                    positions=[x_pos],
                    widths=0.8,
                    patch_artist=True,
                    showfliers=False,
                    whis=[0, 100],
                    boxprops=dict(facecolor=facecolor, alpha=0.6, edgecolor=edgecolor),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(color=edgecolor, linewidth=1.5),
                    capprops=dict(color=edgecolor, linewidth=1.5),
                )

                # Add scatter point for the best subject (if we identified one from models)
                if best_subject is not None:
                    best_subject_sim_data = src_roi_data[
                        src_roi_data["subject"] == best_subject
                    ]
                    if len(best_subject_sim_data) > 0:
                        best_subject_sim_score = best_subject_sim_data["score"].values[0]
                        ax.scatter(
                            x_pos,
                            best_subject_sim_score,
                            color=best_subject_marker_color,
                            edgecolor='black',
                            linewidth=0.5,
                            alpha=0.9,
                            s=50,
                            marker='D',
                            zorder=15
                        )

                all_positions.append(x_pos)
                label = f'{src_roi}\n(self)' if is_self else src_roi
                all_labels.append(label)

        # Update current position for next ROI
        current_pos += n_models_to_plot + n_sim_boxes_plotted

        # # Add text annotation showing top model names below this ROI
        # # (skip if we have too many "best"models)
        # if top_models_list and len(top_models_list) <= 5:
        #     # Shorten model names for readability, but keep provider for CLIP models
        #     short_names = []
        #     for m in top_models_list:
        #         if m.startswith('clip/'):
        #             # Keep CLIP prefix: "clip/ViT-B/32" stays as "clip/ViT-B/32"
        #             short_names.append(m)
        #         elif '/' in m:
        #             # For other models, just show the part after the provider
        #             parts = m.split('/')
        #             short_names.append('/'.join(parts[1:]))
        #         else:
        #             short_names.append(m)

        #     # Format as separate lines (one model per row)
        #     model_text = '\n'.join(short_names)
        #     # Position text at the center of this ROI's section, below the x-axis
        #     # Alternate y-position for every second ROI to avoid overlap
        #     y_offset = -0.07 if roi_idx % 2 == 0 else -0.18
        #     center_pos = base_pos + (n_top_similarities / 2)
        #     ax.text(
        #         center_pos,
        #         y_offset,  # Alternating position
        #         model_text,
        #         ha='center',
        #         va='top',
        #         fontsize=10,
        #         style='italic',
        #         color='darkblue'
        #     )

    # Axis formatting
    ax.set_title(metric)
    ax.set_xticks(all_positions)
    # Use left alignment with rotation so labels start from the tick position
    ax.set_xticklabels(all_labels, rotation=90, ha='center', fontsize=8)
    ax.set_ylabel(f"{metric.replace('_', ' ').title()} Score" +\
         (" Difference" if plot_difference else ""),
                    fontsize=12)
    ax.set_xlabel("", fontsize=12)
    min_y_val = min(top_model_data["score"].min(), top_sim_data["score"].min())
    max_y_val = max(top_model_data["score"].max(), top_sim_data["score"].max())
    delta_y = max_y_val - min_y_val
    ax.set_ylim(min_y_val - 0.30 * delta_y, max_y_val + 0.05 * delta_y)

    #     # Calculate y-limits from all data - Claude's suggestion, don't see why we need to change that
    # all_scores = []
    # if len(top_model_data) > 0:
    #     all_scores.extend(top_model_data["score"].values)
    # if len(top_sim_data) > 0:
    #     all_scores.extend(top_sim_data["score"].values)
    
    # if all_scores:
    #     min_y_val = min(all_scores)
    #     max_y_val = max(all_scores)
    #     delta_y = max_y_val - min_y_val
    #     ax.set_ylim(min_y_val - 0.10 * delta_y, max_y_val + 0.05 * delta_y)

    # ax.set_ylim(-0.30, 1.05)  # Extended lower limit to show staggered model names
    ax.set_xlim(-1, max(all_positions) + 1 if all_positions else 1)
    ax.grid(axis='y', alpha=0.3)

    # Legend
    legend_handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor='steelblue', alpha=0.6, edgecolor='darkblue',
            label=f'Top {n_top_models if n_top_models != -1 else "all"} Models'
        ),
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor='lightcoral', alpha=0.6, edgecolor='darkred',
            label='ROI-to-ROI Similarity'
        ),
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor='gold', alpha=0.6, edgecolor='darkgoldenrod',
            label='Noise Ceiling (self)'
        )
    ]

    # Get handles from the plot (includes the best subject label)
    handles, labels = ax.get_legend_handles_labels()
    # Add the best subject handle if it exists
    if handles:
        legend_handles.extend(handles[:1])  # Just the first one (best subject)

    ax.legend(
        handles=legend_handles,
        loc='upper left',
        frameon=True,
        fontsize=10,
        title='Plot Elements'
    )

    plt.tight_layout()

    # Save plot with reduced DPI for smaller file size
    fname = f"{plot_fname}_{metric}_{split}"
    if plot_difference:
        fname += "_difference"
    save_this(plot_path, fname=fname, formats=formats,
                fig=fig, transparent=False, dpi=100)
    plt.close()

    LOGGER.info(f"Saved combined plot: {fname}")


def _plot_single_roi(
    roi: str,
    model_df: pd.DataFrame,
    intersubject_df: pd.DataFrame,
    subjects: list[str],
    subject_color_map: dict,
    metric: str,
    split: str,
    plot_path: Path,
    formats: tuple[str, ...],
    plot_fname: str,
    n_top_models: int,
    n_top_similarities: int,
    best_subject_marker_color: str = 'darkgreen',
    plot_difference: bool = False,
):
    """Create plot for a single ROI."""
    fig, ax = plt.subplots(figsize=(10, 7))

    all_positions = []
    all_labels = []
    base_pos = 0


    # A. Get top N model scores
    top_model_data = get_top_n_models_per_roi(model_df, roi, n=n_top_models)

    if plot_difference:
        same_same_similarity = get_noise_ceiling(intersubject_df, roi)
        top_model_data = subtract_per_subject(top_model_data, same_same_similarity)

    best_subject = None
    top_models_list = []
    n_models_to_plot = 0
    
    if len(top_model_data) > 0:
        # Get the list of top models for this ROI, ordered by mean score
        model_means = top_model_data.groupby("model")["score"].mean().sort_values(ascending=True)
        top_models_list = model_means.index.tolist()
        n_models_to_plot = len(top_models_list)
        
        # Find the subject with the highest mean model score for this ROI
        subject_means = top_model_data.groupby("subject")["score"].mean()
        best_subject = subject_means.idxmax()

    # Plot each model as a separate boxplot
    if len(top_model_data) > 0:
        for model_idx, model in enumerate(top_models_list):
            x_pos = base_pos + model_idx
            model_data = top_model_data[top_model_data["model"] == model]
            
            # Create boxplot for this model's scores across subjects
            bp = ax.boxplot(
                [model_data["score"].values],
                positions=[x_pos],
                widths=0.8,
                patch_artist=True,
                showfliers=False,
                whis=[0, 100],
                boxprops=dict(facecolor='steelblue', alpha=0.6, edgecolor='darkblue'),
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(color='darkblue', linewidth=1.5),
                capprops=dict(color='darkblue', linewidth=1.5),
            )

            # Add scatter point for the best subject
            best_subject_model_data = model_data[model_data["subject"] == best_subject]
            if len(best_subject_model_data) > 0:
                best_subject_score = best_subject_model_data["score"].values[0]
                ax.scatter(
                    x_pos,
                    best_subject_score,
                    color=best_subject_marker_color,
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=0.9,
                    s=50,
                    zorder=15,
                    label='Best subject' if model_idx == 0 else None
                )
            else:
                LOGGER.warning(
                    f"ROI {roi}: Best subject {best_subject} missing data for model {model}"
                )

            # Shorten model name for label
            if model.startswith('clip/'):
                short_name = model
            elif '/' in model:
                parts = model.split('/')
                short_name = '/'.join(parts[1:])
            else:
                short_name = model
            
            all_positions.append(x_pos)
            all_labels.append(short_name)

    # B. Plot ROI similarities (including self if among top)
    top_sim_data = get_top_n_roi_similarities(
        intersubject_df, roi, n=n_top_similarities, include_self=True
    )
    if plot_difference:
        top_sim_data = subtract_per_subject(top_sim_data, same_same_similarity)

    if len(top_sim_data) > 0:
        source_rois = (
            top_sim_data.groupby("layer")["score"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )

        for idx, src_roi in enumerate(source_rois):
            x_pos = base_pos + n_models_to_plot + idx
            src_roi_data = top_sim_data[top_sim_data["layer"] == src_roi]

            is_self = src_roi == roi
            facecolor = 'gold' if is_self else 'lightcoral'
            edgecolor = 'darkgoldenrod' if is_self else 'darkred'

            ax.boxplot(
                [src_roi_data["score"].values],
                positions=[x_pos],
                widths=0.8,
                patch_artist=True,
                showfliers=False,
                whis=[0, 100],
                boxprops=dict(facecolor=facecolor, alpha=0.6, edgecolor=edgecolor),
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(color=edgecolor, linewidth=1.5),
                capprops=dict(color=edgecolor, linewidth=1.5),
            )

            # Add scatter point for the best subject
            if best_subject is not None:
                best_subject_sim_data = src_roi_data[
                    src_roi_data["subject"] == best_subject
                ]
                if len(best_subject_sim_data) > 0:
                    best_subject_sim_score = best_subject_sim_data["score"].values[0]
                    ax.scatter(
                        x_pos,
                        best_subject_sim_score,
                        color=best_subject_marker_color,
                        edgecolor='black',
                        linewidth=0.5,
                        alpha=0.9,
                        s=50,
                        marker='D',
                        zorder=15
                    )

            all_positions.append(x_pos)
            label = f'{src_roi}\n(self)' if is_self else src_roi
            all_labels.append(label)

    # Axis formatting
    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel(f"{metric.replace('_', ' ').title()} Score" +\
         (" Difference" if plot_difference else ""),
                    fontsize=12)
    ax.set_xlabel("", fontsize=12)
    min_y_val = min(top_model_data["score"].min(), top_sim_data["score"].min())
    max_y_val = max(top_model_data["score"].max(), top_sim_data["score"].max())
    delta_y = max_y_val - min_y_val
    ax.set_ylim(min_y_val - 0.05 * delta_y, max_y_val + 0.05 * delta_y)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-1, max(all_positions) + 1 if all_positions else 1)
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(f"ROI: {roi}"+(" (difference)" if plot_difference else ""),
                 fontsize=14, weight='bold')

    # Legend
    legend_handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor='steelblue', alpha=0.6, edgecolor='darkblue',
            label=f'Top {n_top_models if n_top_models != -1 else "all"} Models'
        ),
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor='lightcoral', alpha=0.6, edgecolor='darkred',
            label='ROI-to-ROI Similarity'
        ),
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor='gold', alpha=0.6, edgecolor='darkgoldenrod',
            label='Noise Ceiling (self)'
        )
    ]

    # Get handles from the plot (includes the best subject label)
    handles, labels = ax.get_legend_handles_labels()
    # Add the best subject handle if it exists
    if handles:
        legend_handles.extend(handles[:1])  # Just the first one (best subject)

    ax.legend(
        handles=legend_handles,
        loc='lower center',
        frameon=True,
        fontsize=10,
        title='Plot Elements'
    )

    plt.tight_layout()

    # Save plot with reduced DPI for smaller file size
    fname = f"{plot_fname}_{roi}_{metric}_{split}"
    if plot_difference:
        fname += "_difference"
    save_this(plot_path, fname=fname, formats=formats, fig=fig,
                transparent=False, dpi=100)
    plt.close()

    LOGGER.info(f"Saved plot for ROI {roi}: {fname}")


# @click.option(
#     "--filename",
#     type=str,
#     default="scores_rsa.csv",
#     help="DEPRECATED, use --metric instead. Name of scoresheet files to load. Typically 'scores_rsa.csv' "
#          "or 'scores_linear_predictivity.csv'."
# )

@click.command()
@click.option(
    "--model-results-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing model benchmark results (scoresheets)."
)
@click.option(
    "--intersubject-results-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing intersubject consistency results "
         "(ROI-to-ROI similarity scoresheets)."
)
@click.option(
    "--split",
    type=str,
    default="test",
    help="Which split to plot: 'train' or 'test'."
)
@click.option(
    "--metric",
    type=str,
    default="rsa",
    help="Metric to plot: 'rsa' or 'linear_predictivity'."
)
@click.option(
    "--plot-path",
    type=click.Path(exists=False, file_okay=False, path_type=Path),
    required=True,
    help="Output directory for plots."
)
@click.option(
    "--n-top-models",
    type=int,
    default=3,
    help="Number of top-scoring models to include per ROI. Use -1 for all models."
)
@click.option(
    "--n-top-similarities",
    type=int,
    default=10,
    help="Number of top ROI-to-ROI similarities to show per ROI."
)
@click.option(
    "--format",
    "formats",
    type=click.Choice(["pdf", "png", "svg"], case_sensitive=False),
    multiple=True,
    default=['png'],
    help="Output format(s) for the plot."
)
@click.option(
    "--plot-fname",
    type=str,
    default="roi_best_models_with_brain_sim",
    help="Base filename for saved plot."
)
@click.option(
    "--per-roi-plots",
    is_flag=True,
    default=False,
    help="If set, create separate plots for each ROI instead of a single combined plot."
)
@click.option(
    "--roi-list-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to YAML config file specifying which ROIs to include (e.g., rois_1b.yaml). "
         "If not provided, all ROIs will be included."
)
@click.option(
    "--plot-difference",
    is_flag=True,
    default=False,
    help=("Plot the difference between same-same similarity "
        "and same-other similarity, instead of raw values.")
)
def plot_best_with_brain_sim(
    model_results_dir: Path,
    intersubject_results_dir: Path,
    # filename: str,
    split: str,
    metric: str,
    plot_path: Path,
    n_top_models: int,
    n_top_similarities: int,
    formats: tuple[str, ...],
    plot_fname: str,
    per_roi_plots: bool,
    roi_list_config: Path | None,
    plot_difference: bool,
):
    """Plot top model scores with brain-brain similarities, color-coded by subject.
    
    Creates visualizations showing:
    - Top N model scores for each ROI (box plots + subject-colored points)
    - Top M ROI-to-ROI similarities (subject-colored diamond markers)
    - Noise ceiling for each ROI
    
    Key feature: Subject-level color coding enables visual matching of scores
    from the same subject across different comparison types.
    """
    # Create output directory
    plot_path.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("=" * 60)
    LOGGER.info("Starting plot generation")
    LOGGER.info(f"Model results: {model_results_dir}")
    LOGGER.info(f"Intersubject results: {intersubject_results_dir}")
    LOGGER.info(f"Metric: {metric}, Split: {split}")
    LOGGER.info(f"Top models: {n_top_models}, Top similarities: {n_top_similarities}")
    LOGGER.info(f"Per-ROI plots: {per_roi_plots}")
    if roi_list_config:
        LOGGER.info(f"ROI list config: {roi_list_config}")
    LOGGER.info("=" * 60)

    # Load ROI list from config if provided
    roi_subset = None
    if roi_list_config:
        with open(roi_list_config, 'r') as f:
            config = yaml.safe_load(f)
        roi_subset = config.get('data', {}).get('rois', None)
        if roi_subset:
            LOGGER.warning(f"Filtering to {len(roi_subset)} ROIs from config: {roi_subset}")
        else:
            LOGGER.warning(f"No ROIs found in config file at data.rois, using all ROIs")
    
    # Load data
    model_df = load_model_results(
        model_results_dir, #filename, 
        split, metric
    )
    intersubject_df = load_intersubject_similarities(
        intersubject_results_dir, #filename, 
        split, metric
    )
    
    # Validate ROI consistency
    model_rois = set(model_df["roi"].unique())
    intersubject_rois = set(intersubject_df["roi"].unique())
    
    common_rois = model_rois & intersubject_rois
    if not common_rois:
        raise ValueError(
            "No common ROIs found between model and intersubject results!"
        )
    
    if model_rois - common_rois:
        LOGGER.warning(
            f"ROIs in model results but not in intersubject: "
            f"{model_rois - common_rois}"
        )
    if intersubject_rois - common_rois:
        LOGGER.warning(
            f"ROIs in intersubject but not in model results: "
            f"{intersubject_rois - common_rois}"
        )
    
    # Order ROIs by hierarchy
    roi_order_df = model_df[["roi", "stream", "order"]].drop_duplicates()
    roi_order_df = roi_order_df.sort_values(["stream", "order", "roi"])
    rois = [roi for roi in roi_order_df["roi"].unique() if roi in common_rois]

    # Filter to ROI subset if config was provided
    if roi_subset is not None:
        original_roi_count = len(rois)
        missing_from_data = set(roi_subset) - set(rois)
        rois = [roi for roi in rois if roi in roi_subset]
        if len(rois) == 0:
            raise ValueError(
                f"None of the ROIs from config ({roi_subset}) are available in the data. "
                f"Available ROIs: {list(common_rois)}"
            )
        LOGGER.warning(f"Filtered from {original_roi_count} to {len(rois)} ROIs based on config")
        if missing_from_data:
            LOGGER.warning(f"ROIs in config but not available in data: {sorted(missing_from_data)}")

    LOGGER.warning(f"Plotting {len(rois)} ROIs: {rois}")
    
    # Create plots
    create_roi_comparison_plot(
        model_df,
        intersubject_df,
        rois,
        metric,
        split,
        plot_path,
        formats,
        plot_fname,
        n_top_models,
        n_top_similarities,
        per_roi_plots,
        plot_difference,
    )
    
    LOGGER.info("=" * 60)
    LOGGER.info("Plot generation complete!")
    LOGGER.info(f"Plots saved to: {plot_path}")
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    plot_best_with_brain_sim()

