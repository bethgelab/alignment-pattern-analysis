#!/usr/bin/env python3
"""Script to analyze alignment pattern similarity for random initialization models.

This script performs alignment pattern similarity analysis comparing trained vs
untrained (random initialization) models. It was converted from a Jupyter notebook.
"""

import logging
import pickle as pkl
from collections import defaultdict
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multitasking.analyses.alignment_patterns.model_brain_alignment_patterns import (
    ModelBrainAlignmentPatterns,
)
from multitasking.plot_creation.alignment_pattern_plots import (
    plot_alignment_patterns_per_roi_simple,
    plot_trained_untrained_comparison,
)
from multitasking.plot_creation.dataframe_utils import (
    get_order,
    load_bm_results,
    select_top_layer,
)

LOGGER = logging.getLogger(__name__)

# Define ROI assignments
roi_stream = {
    "early": ["V1", "V2", "V3"],
    "ventral": ["V4", "V8", "PIT", "FFC"],
    "dorsal": ["PH", "V6", "V7", "IPS1", "MT", "MST", "FST", "LO1", "LO2", "LO3", "V3A", "V3B", "V6A"]
}


def classify_stream(region: str) -> str:
    """Classify a region into a visual stream."""
    if region in roi_stream['early']:
        return 'early'
    elif region in roi_stream['ventral']:
        return 'ventral'
    elif region in roi_stream['dorsal']:
        return 'dorsal'
    return 'unknown'


@click.command()
@click.option(
    '--split',
    default='test',
    help='Data split to use (default: test)',
    type=str,
)
@click.option(
    '--output-supdir',
    required=True,
    help='Path to BM results directory for random init models',
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    '--trained-model-aps-path',
    required=True,
    help='Path to trained model-brain alignment patterns pickle file',
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    '--brain-brain-aps-path',
    required=True,
    help='Path to brain-brain alignment patterns pickle file',
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    '--analysis-results-path',
    required=True,
    help='Path to equivalent_models pickle file',
    type=click.Path(exists=True, path_type=str),
)
@click.option(
    '--metric',
    default='rsa',
    help='Metric to use (default: rsa)',
    type=click.Choice(['rsa', 'linear_predictivity']),
)
@click.option(
    '--plot-path',
    required=True,
    help='Path to directory where plots will be saved',
    type=click.Path(path_type=Path),
)
@click.option(
    '--model-colors-dict-path',
    default=None,
    help='Path to model colors dictionary pickle file. If not provided, will be loaded from default location.',
    type=click.Path(path_type=Path),
)
@click.option(
    '--rois',
    default='V1,V2,V3,V4,V8,PIT,FFC,V3A,V3B,V6,V6A,V7,IPS1,MT,MST,FST,LO1,LO2,LO3',
    help='Comma-separated list of ROIs to analyze',
    type=str,
)
@click.option(
    '--save-results/--no-save-results',
    default=True,
    help='Whether to save results to pickle files',
)
@click.option(
    '--save-plots/--no-save-plots',
    default=True,
    help='Whether to save plots',
)
@click.option(
    '--log-level',
    default='INFO',
    help='Logging level',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
)
@click.option(
    '--similarity-metric',
    default='pearson',
    help='Similarity metric to use for alignment pattern similarity (default: pearson)',
    type=click.Choice(['pearson', 'mse']),
)
def main(
    split: str,
    output_supdir: Path,
    trained_model_aps_path: Path,
    brain_brain_aps_path: Path,
    analysis_results_path: str,
    plot_path: Path,
    model_colors_dict_path: Path,
    rois: str,
    metric: str,
    save_results: bool,
    save_plots: bool,
    log_level: str,
    similarity_metric: str,
):
    """Run alignment pattern similarity analysis for random initialization models."""
    # Set up logging
    logging.basicConfig(level=getattr(logging, log_level))

    # Parse ROIs
    roi_list = [r.strip() for r in rois.split(',')]
    LOGGER.info(f"Analyzing ROIs: {roi_list}")

    # Create plot directory
    plot_path.mkdir(parents=True, exist_ok=True)

    # Load trained model and brain-brain alignment patterns
    LOGGER.info("Loading trained model and brain-brain alignment patterns...")
    with open(trained_model_aps_path, 'rb') as f:
        trained_model_brain_alignment_patterns_rsa = pkl.load(f)

    with open(brain_brain_aps_path, 'rb') as f:
        pw_brain_brain_alignment_patterns_rsa = pkl.load(f)

    # Load and organize random init results
    LOGGER.info("Loading BM results for random init models...")
    full_df = load_bm_results(output_supdir)

    # Add stream information
    full_df.insert(full_df.shape[1], "stream", full_df["roi"].apply(classify_stream))

    # Subject average
    subject_avg_df = full_df.groupby(
        ["model", "layer", "roi", "split", "metric", "stream"], as_index=False
    ).agg(
        count=("score", "count"),
        score_mean=("score", "mean"),
        score_std=("score", "std"),
    )

    # Select best layer for each model and roi
    LOGGER.info("Selecting best layers...")
    df, roi_layer_map = select_top_layer(subject_avg_df, split="train")
    assert isinstance(roi_layer_map, pd.DataFrame), "roi_layer_map should be a DataFrame"

    # Add boolean column indicating best layer
    roi_layer_map.insert(roi_layer_map.shape[1], 'is_best_layer', True)
    full_df = full_df.merge(roi_layer_map, on=["roi", "layer", "metric", "model"], how="outer")
    boolean_layer = full_df['is_best_layer'].apply(lambda x: False if np.isnan(x) else x)
    full_df.drop("is_best_layer", axis=1, inplace=True)
    full_df.insert(full_df.shape[1], 'is_best_layer', boolean_layer)

    # Get model ordering
    LOGGER.info("Computing model orders...")
    model_orders = {}
    for _metric in ['rsa', 'linear_predictivity']:
        model_orders[f"{_metric}"] = get_order(
            df,
            metric=_metric,
            order_by="model"
        )
        df[f"model_order_{_metric}"] = pd.Categorical(
            df["model"],
            categories=model_orders[f"{_metric}"],
            ordered=True
        )
    # Load equivalent models
    LOGGER.info(f"Loading equivalent models from {analysis_results_path}...")
    equivalent_models_df = pkl.load(
        open(analysis_results_path, "rb")
    )

    unique_model_layers = list(set(zip(
        equivalent_models_df.query(f"metric == '{metric}'")['model'],
        equivalent_models_df.query(f"metric == '{metric}'")['layer'], strict=False
    )))
    unique_model_layers.sort()

    unique_model_layers = [(model, layer) for model, layer in unique_model_layers \
                             if full_df.query(f'model == "{model}" & layer == "{layer}"').shape[0] > 0]

    unique_model_layers_names = [f"{model}__{layer}" for model, layer in unique_model_layers]
    LOGGER.info(f"Found {len(unique_model_layers_names)} unique model-layer combinations")

    # Build dictionaries
    model_layer_roi_dict: dict[str, list[str]] = {ml: [] for ml in unique_model_layers_names}

    for model, layer in unique_model_layers:
        ml_rois = equivalent_models_df.query(
            f"metric == '{metric}' & model =='{model}' & layer == '{layer}'"
        )["roi"].to_list()
        for r in ml_rois:
            model_layer_roi_dict[f"{model}__{layer}"].append(r)

    roi_model_layer_dict: dict[str, list[str]] = {r: [] for r in roi_list}
    for model_layer, _rois in model_layer_roi_dict.items():
        for r in _rois:
            roi_model_layer_dict[r].append(model_layer)
    roi_model_layer_dict = dict(roi_model_layer_dict)

    # Create dict for all model layers (not just those matching this ROI)
    all_all_roi_model_layer_dict: dict[str, list[str]] = defaultdict(list)
    for model_layer in model_layer_roi_dict.keys():
        for r in roi_list:
            all_all_roi_model_layer_dict[r].append(model_layer)
    all_all_roi_model_layer_dict = dict(all_all_roi_model_layer_dict)

    # Model-brain alignment patterns for random init
    LOGGER.info("Computing model-brain alignment patterns for random init...")
    model_brain_alignment_patterns_rsa = ModelBrainAlignmentPatterns(
        predictor_type="model",
        predictor_names=unique_model_layers_names,
        target_type="brain",
        target_names=roi_list,
        metric=metric
    )

    model_brain_alignment_patterns_rsa.get_alignment_pattern(full_df, split=split)

    # Get subjects and reference patterns
    subjects = list(pw_brain_brain_alignment_patterns_rsa.subject_predictor_mapping.keys())

    result = pw_brain_brain_alignment_patterns_rsa.get_alignment_pattern_similarity(
        similarity_metric=similarity_metric,
        on="exclude_subject",
        subjects=subjects,
    )
    # Unpack the tuple with proper type annotations
    # Type: tuple[dict[str, dict[str, list[float]]], dict[str, list[float]], dict[str, dict[str, list[float]]]]
    _, alignment_pattern_similarity_loo, reference_patterns_by_subject_raw = result
    # Convert reference_patterns_by_subject from dict[str, dict[str, list[float]]]
    # to dict[str, dict[str, np.ndarray]] for ModelBrainAlignmentPatterns
    reference_patterns_by_subject: dict[str, dict[str, np.ndarray]] = {
        subject: {
            roi: np.array(pattern)
            for roi, pattern in patterns.items()
        }
        for subject, patterns in reference_patterns_by_subject_raw.items()
    }

    # Get alignment pattern similarity for random init models
    # Type annotation says dict[str, np.ndarray] but when subjects is provided,
    # it actually accepts dict[str, dict[str, np.ndarray]]
    model_aps = model_brain_alignment_patterns_rsa.get_alignment_pattern_similarity(
        similarity_metric=similarity_metric,
        roi_model_layer_dict=roi_model_layer_dict,
        subjects=subjects,
        reference_alignment_patterns=reference_patterns_by_subject  # type: ignore[arg-type]
    )

    # Get alignment pattern similarity for trained models
    trained_model_aps = trained_model_brain_alignment_patterns_rsa.get_alignment_pattern_similarity(
        similarity_metric=similarity_metric,
        roi_model_layer_dict=roi_model_layer_dict,
        subjects=subjects,
        reference_alignment_patterns=reference_patterns_by_subject  # type: ignore[arg-type]
    )

    # Create plots
    if save_plots:
        LOGGER.info("Creating plots...")

        # Load or create model colors
        if model_colors_dict_path is not None and model_colors_dict_path.exists():
            LOGGER.info(f"Loading model colors from {model_colors_dict_path}")
            with open(model_colors_dict_path, 'rb') as f:
                model_colors_dict = pkl.load(f)
        else:
            LOGGER.info("Generating model colors...")
            cmap = plt.get_cmap("tab20", len(unique_model_layers_names))
            model_colors_dict = {model_layer: cmap(i) for i, model_layer in enumerate(unique_model_layers_names)}

        # Get ROI colormap
        similarities = alignment_pattern_similarity_loo

        # Plot 1: Overview with trained vs untrained side by side
        LOGGER.info("Creating overview plot (trained vs untrained side by side)...")
        plot_trained_untrained_comparison(
            brain_brain_aps=similarities,
            trained_model_brain_aps=trained_model_aps,
            model_brain_aps=model_aps,
            roi_list=roi_list,
            model_colors_dict=model_colors_dict,
            plot_path=plot_path,
            metric=metric,
            roi_model_layer_dict=roi_model_layer_dict,
        )

        # Plot 2: Overview with trained vs untrained overlaid
        # LOGGER.info("Creating overview plot (trained vs untrained overlaid)...")
        # x_pos = 0
        # xtick_positions = []
        # fig, axes = plt.subplots(
        #     2, 1,
        #     gridspec_kw={'height_ratios': [1, 0.5], "hspace": 0.5},
        #     figsize=(12, 8)
        # )

        # roi_spacing = 2  # space between ROIs

        # flierprops_roi = dict(
        #     marker='o',
        #     markersize=3,
        #     markerfacecolor='black',
        #     markeredgecolor='black',
        #     alpha=0.6
        # )

        # for i, roi in enumerate(roi_list):
        #     # --- Within-ROI boxplot ---
        #     within_scores = similarities[roi]

        #     axes[0].boxplot(
        #         within_scores,
        #         positions=[x_pos],
        #         widths=0.6,
        #         patch_artist=True,
        #         boxprops=dict(facecolor='black', edgecolor='black'),
        #         medianprops=dict(color='white'),
        #         whiskerprops=dict(color='black'),
        #         capprops=dict(color='black'),
        #         flierprops=flierprops_roi
        #     )

        #     x_pos += 1  # move to model layers for this ROI

        #     # --- Model layers boxplots ---
        #     n_models = len(roi_model_layer_dict[roi])
        #     x_positions = np.linspace(x_pos, x_pos + n_models - 1, n_models)

        #     sorted_models = sorted(
        #         [(k, v) for k, v in model_aps[roi].items() if k in roi_model_layer_dict[roi]],
        #         key=lambda kv: np.mean(kv[1]),
        #         reverse=True
        #     )

        #     for j, (model_layer, aps_scores) in enumerate(sorted_models):
        #         flierprops_model = dict(
        #             marker='^',
        #             markersize=2,
        #             markerfacecolor=model_colors_dict[model_layer],
        #             markeredgecolor=model_colors_dict[model_layer],
        #             alpha=0.7
        #         )

        #         # Untrained
        #         axes[0].boxplot(
        #             aps_scores,
        #             positions=[x_positions[j]],
        #             widths=0.6,
        #             patch_artist=True,
        #             boxprops=dict(facecolor='white',
        #                           edgecolor=model_colors_dict[model_layer]),
        #             medianprops=dict(color='black'),
        #             whiskerprops=dict(color=model_colors_dict[model_layer]),
        #             capprops=dict(color=model_colors_dict[model_layer]),
        #             flierprops=flierprops_model
        #         )

        #         # Trained
        #         trained_aps_scores = trained_model_aps[roi][model_layer]

        #         axes[0].boxplot(
        #             trained_aps_scores,
        #             positions=[x_positions[j]],
        #             widths=0.6,
        #             patch_artist=True,
        #             boxprops=dict(facecolor=model_colors_dict[model_layer],
        #                           edgecolor=model_colors_dict[model_layer]),
        #             medianprops=dict(color='black'),
        #             whiskerprops=dict(color=model_colors_dict[model_layer]),
        #             capprops=dict(color=model_colors_dict[model_layer]),
        #             flierprops=flierprops_model
        #         )

        #     # --- shading behind this ROI group ---
        #     axes[0].fill_betweenx(
        #         [-0.6, 1],
        #         x_pos - 1,
        #         x_positions[-1] + 1,
        #         color='grey',
        #         alpha=0.1
        #     )

        #     # tick at ROI center
        #     xtick_positions.append((x_pos - 1 + x_positions[-1]) / 2)

        #     # update x position for next ROI
        #     x_pos = x_positions[-1] + roi_spacing

        # # --- Axis labels ---
        # axes[0].set_xticks(xtick_positions)
        # axes[0].set_xticklabels(roi_list, rotation=90)

        # # --- Legend Panel ---
        # legend_elements = [
        #     Patch(facecolor=model_colors_dict[model_layer],
        #           label=model_layer[:40])
        #     for model_layer in unique_model_layers_names
        # ]
        # legend_elements.insert(0,
        #        Patch(facecolor='k',
        #       label=f"subject-subject"))

        # axes[1].legend(
        #     handles=legend_elements,
        #     loc='center',
        #     fontsize=9,
        #     title='Model-Layer Combinations',
        #     title_fontsize=11,
        #     frameon=True,
        #     fancybox=True,
        #     shadow=True
        # )
        # axes[0].axhline(0, color='k', linestyle='--', alpha=0.2, zorder=0)
        # axes[1].axis('off')
        # sns.despine()
        # axes[0].set_ylabel("Alignment pattern similarity \n (Pearson R)")
        # axes[0].set_xlabel("ROIs")

        # save_this(plot_path, fname=f"alignment_pattern_overview_{metric}_trained_untrained",
        #           formats=['svg', 'png', 'pdf'], transparent=False)
        # plt.close()

        # # Plot 3: Per model-layer combination
        # LOGGER.info("Creating per-model plots...")
        # for model_layer in unique_model_layers_names:
        #     x_pos = 0
        #     xtick_positions = []

        #     fig, axes = plt.subplots(
        #         2, 1,
        #         gridspec_kw={'height_ratios': [1, 0.5], "hspace": 0.5},
        #         figsize=(12, 8)
        #     )

        #     roi_spacing = 1.5

        #     flierprops_roi = dict(
        #         marker='o',
        #         markersize=3,
        #         markerfacecolor='black',
        #         markeredgecolor='black',
        #         alpha=0.6
        #     )

        #     flierprops_untrained = dict(
        #         marker='^',
        #         markersize=3,
        #         markerfacecolor=model_colors_dict[model_layer],
        #         markeredgecolor=model_colors_dict[model_layer],
        #         alpha=0.7
        #     )

        #     flierprops_trained = dict(
        #         marker='s',
        #         markersize=3,
        #         markerfacecolor='white',
        #         markeredgecolor=model_colors_dict[model_layer],
        #         alpha=0.7
        #     )

        #     for i, roi in enumerate(roi_list):
        #         # Within ROI
        #         within_scores = similarities[roi]

        #         axes[0].boxplot(
        #             within_scores,
        #             positions=[x_pos],
        #             widths=0.6,
        #             patch_artist=True,
        #             boxprops=dict(facecolor="black", edgecolor="black"),
        #             medianprops=dict(color='white'),
        #             whiskerprops=dict(color='black'),
        #             capprops=dict(color='black'),
        #             flierprops=flierprops_roi
        #         )

        #         x_pos += 1

        #         # Untrained model
        #         untrained_scores = model_aps[roi][model_layer]

        #         axes[0].boxplot(
        #             untrained_scores,
        #             positions=[x_pos],
        #             widths=0.6,
        #             patch_artist=True,
        #             boxprops=dict(facecolor="white",
        #                           edgecolor=model_colors_dict[model_layer]),
        #             medianprops=dict(color='black'),
        #             whiskerprops=dict(color=model_colors_dict[model_layer]),
        #             capprops=dict(color=model_colors_dict[model_layer]),
        #             flierprops=flierprops_untrained
        #         )

        #         x_pos += 1

        #         # Trained model
        #         trained_scores = trained_model_aps[roi][model_layer]

        #         axes[0].boxplot(
        #             trained_scores,
        #             positions=[x_pos],
        #             widths=0.6,
        #             patch_artist=True,
        #             boxprops=dict(facecolor=model_colors_dict[model_layer],
        #                           edgecolor=model_colors_dict[model_layer],
        #                           linewidth=1.5),
        #             medianprops=dict(color='black'),
        #             whiskerprops=dict(color=model_colors_dict[model_layer]),
        #             capprops=dict(color=model_colors_dict[model_layer]),
        #             flierprops=flierprops_trained
        #         )

        #         # Shaded background for this ROI
        #         axes[0].fill_betweenx(
        #             [-0.6, 1],
        #             x_pos - 2,
        #             x_pos + 1,
        #             color='grey',
        #             alpha=0.08
        #         )

        #         # Tick in the center of this ROI group
        #         xtick_positions.append(x_pos - 1)

        #         # Next ROI
        #         x_pos += roi_spacing

        #     # Axis formatting
        #     axes[0].set_xticks(xtick_positions)
        #     axes[0].set_xticklabels(roi_list, rotation=90)

        #     axes[0].axhline(0, color='k', linestyle='--', alpha=0.2)
        #     axes[0].set_ylabel("Alignment Pattern Similarity (Pearson r)")
        #     axes[0].set_xlabel("ROIs")
        #     axes[0].set_title(model_layer)

        #     axes[1].axis('off')
        #     sns.despine()

        #     save_this(plot_path, fname=f'trained_untrained_{model_layer.replace("/", "_")}',
        #               transparent=False)
        #     plt.close()

        # Plot 4: Alignment patterns per ROI
        LOGGER.info("Creating per-ROI alignment pattern plots...")
        for roi in roi_list:
            plot_alignment_patterns_per_roi_simple(
                roi=roi,
                roi_list=roi_list,
                brain_brain_aps=similarities,
                model_brain_aps=model_aps,
                model_colors_dict=model_colors_dict,
                plot_path=plot_path,
                metric=metric,
                pw_brain_brain_alignment_patterns=pw_brain_brain_alignment_patterns_rsa,
                model_brain_alignment_patterns=model_brain_alignment_patterns_rsa,
                roi_model_layer_dict=roi_model_layer_dict,
            )

        # # Plot 5: Alignment patterns with scatter
        # LOGGER.info("Creating alignment pattern plots with scatter...")
        # for roi in roi_list:
        #     fig, axes = plt.subplots(
        #         1, 2,
        #         figsize=(8, 4),
        #         gridspec_kw={'width_ratios': [2, 1], "wspace": 0.3}
        #     )

        #     # Left panel: shading
        #     brain_ap_matrix = pw_brain_brain_alignment_patterns_rsa.alignment_pattern_dict_by_roi[roi]
        #     brain_mean = brain_ap_matrix.mean(axis=1)
        #     brain_std = brain_ap_matrix.std(axis=1)

        #     axes[0].fill_between(range(len(roi_list)), brain_mean - brain_std, brain_mean + brain_std,
        #                         color='k', alpha=0.1)
        #     axes[0].plot(range(len(roi_list)), brain_mean, color='k', linewidth=2)

        #     # Models
        #     for model_layer in roi_model_layer_dict[roi]:
        #         subj_dict = model_brain_alignment_patterns_rsa.alignment_pattern_per_subject[model_layer]
        #         model_mat = np.array([[model_ap[_roi] for _roi in roi_list]
        #                              for _, model_ap in subj_dict.items()]).T
        #         model_mean = model_mat.mean(axis=1)
        #         model_std = model_mat.std(axis=1)
        #         color = model_colors_dict[model_layer]

        #         axes[0].fill_between(range(len(roi_list)), model_mean - model_std, model_mean + model_std,
        #                             color=color, alpha=0.15)
        #         axes[0].plot(range(len(roi_list)), model_mean, color=color, linewidth=2)

        #     axes[0].set_xticks(range(len(roi_list)))
        #     axes[0].set_xticklabels(roi_list, rotation=90)
        #     axes[0].set_xlabel("Target ROIs")
        #     axes[0].set_ylabel("Alignment score (RSA)")
        #     axes[0].set_title(f"{roi}: Mean ± std alignment patterns")

        #     # Right panel: scatter for this ROI
        #     x_pos = 0
        #     scatter_width = 0.15
        #     x_labels = []

        #     # Brain
        #     brain_vals = similarities[roi]
        #     jitter = np.random.uniform(-scatter_width, scatter_width, size=len(brain_vals))
        #     axes[1].scatter(
        #         np.ones_like(brain_vals) * x_pos + jitter,
        #         brain_vals,
        #         color='k',
        #         alpha=0.6,
        #     )
        #     x_labels.append(roi)
        #     x_pos += 1

        #     # Models
        #     for model_layer in roi_model_layer_dict[roi]:
        #         model_vals = model_aps[roi][model_layer]
        #         jitter = np.random.uniform(-scatter_width, scatter_width, size=len(model_vals))
        #         axes[1].scatter(
        #             np.ones_like(model_vals) * x_pos + jitter,
        #             model_vals,
        #             color=model_colors_dict[model_layer],
        #             alpha=0.6,
        #         )
        #         x_labels.append(model_layer[:20])
        #         x_pos += 1

        #     axes[1].set_xlabel("Predictor feature spaces")
        #     axes[1].set_xticks(range(x_pos))
        #     axes[1].set_xticklabels(x_labels, rotation=90)
        #     axes[1].set_ylabel("Alignment similarity (Pearson R)")
        #     axes[1].set_title(f"Alignment Pattern Similarities")

        #     sns.despine()
        #     save_this(plot_path, fname=f"{metric}_ap_{roi}", transparent=False)
        #     plt.close()

    # Save results
    if save_results:
        LOGGER.info("Saving results...")
        results_dir = plot_path.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        with open(results_dir / f'model_brain_aps_{metric}_random_init.pkl', 'wb') as f:
            pkl.dump(model_brain_alignment_patterns_rsa, f)

        LOGGER.info(f"Results saved to {results_dir}")

    LOGGER.info("Analysis complete!")


if __name__ == '__main__':
    main()

