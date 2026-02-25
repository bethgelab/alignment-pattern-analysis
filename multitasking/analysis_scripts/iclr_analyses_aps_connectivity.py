#!/usr/bin/env python3
"""Script to analyze alignment pattern similarity with connectivity-based patterns.

This script performs alignment pattern similarity analysis comparing brain-brain
and model-brain alignment patterns with connectivity-based reference patterns.
It was converted from a Jupyter notebook.
"""

import logging
import pickle as pkl
from collections import defaultdict
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

from multitasking.plot_creation.alignment_pattern_plots import (
    plot_connectivity_alignment_pattern_overview,
    plot_connectivity_alignment_pattern_per_roi,
    plot_random_connectivity_comparison,
)

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    '--trained-model-aps-path',
    required=True,
    help='Path to trained model-brain APS pickle file',
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    '--brain-brain-aps-path',
    required=True,
    help='Path to brain-brain APS pickle file',
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    '--brain-brain-aps-other-metric',
    required=True,
    help='Path to brain-brain APS pickle file for other metric',
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    '--connectivity-aps-path',
    required=True,
    help='Path to connectivity APS pickle file',
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    '--analysis-results-path',
    required=True,
    help='Path to equivalent_models_281125.pkl',
    type=click.Path(exists=True, path_type=str),
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
    help='Path to model colors dictionary pickle file. If not provided, will be generated.',
    type=click.Path(path_type=Path),
)
@click.option(
    '--metric',
    default='rsa',
    help='Metric to use (default: rsa)',
    type=click.Choice(['rsa', 'linear_predictivity']),
)
@click.option(
    '--other-metric',
    default='linear_predictivity',
    help='Other metric to use (default: linear_predictivity)',
    type=click.Choice(['rsa', 'linear_predictivity']),
)
@click.option(
    '--rois',
    default='V1,V2,V3,V4,V8,PIT,FFC,V3A,V3B,V6,V6A,V7,IPS1,MT,MST,FST,LO1,LO2,LO3',
    help='Comma-separated list of ROIs to analyze',
    type=str,
)
@click.option(
    '--random-connectivity-aps-path',
    default=None,
    help='Path to random connectivity APS pickle file (optional, for null distribution analysis)',
    type=click.Path(exists=True, path_type=Path),
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
    trained_model_aps_path: Path,
    brain_brain_aps_path: Path,
    brain_brain_aps_other_metric: Path,
    connectivity_aps_path: Path,
    analysis_results_path: str,
    plot_path: Path,
    model_colors_dict_path: Path,
    metric: str,
    other_metric: str,
    rois: str,
    random_connectivity_aps_path: Path,
    save_plots: bool,
    log_level: str,
    similarity_metric: str,
):
    """Run alignment pattern similarity analysis with connectivity-based patterns."""
    # Set up logging
    logging.basicConfig(level=getattr(logging, log_level))

    # Parse ROIs
    roi_list = [r.strip() for r in rois.split(',')]
    LOGGER.info(f"Analyzing ROIs: {roi_list}")

    # Create plot directory
    plot_path.mkdir(parents=True, exist_ok=True)

    # Load trained model and brain-brain APS
    LOGGER.info("Loading trained model-brain APS...")
    with open(trained_model_aps_path, 'rb') as f:
        trained_model_brain_alignment_patterns = pkl.load(f)

    LOGGER.info("Loading brain-brain APS...")
    with open(brain_brain_aps_path, 'rb') as f:
        pw_brain_brain_alignment_patterns = pkl.load(f)

    LOGGER.info("Loading brain-brain APS for other metric...")
    with open(brain_brain_aps_other_metric, 'rb') as f:
        pw_brain_brain_alignment_patterns_other_metric = pkl.load(f)

    # Load connectivity-based APs
    LOGGER.info("Loading connectivity-based APS...")
    with open(connectivity_aps_path, "rb") as f:
        connectivity_aps_meta = pkl.load(f)
    connectivity_rois = connectivity_aps_meta["rois"]
    connectivity_aps = connectivity_aps_meta["connectivity_aps"]

    # Build connectivity-based alignment pattern dictionary
    connectivity_based_ap = {
        roi: {
            target_roi: connectivity_aps[:, connectivity_rois.index(roi), connectivity_rois.index(target_roi)].mean()
            for target_roi in connectivity_rois if target_roi != roi
        }
        for roi in connectivity_rois
    }

    connectivity_based_ap_std = {
        roi: {
            target_roi: connectivity_aps[:, connectivity_rois.index(roi), connectivity_rois.index(target_roi)].std()
            for target_roi in connectivity_rois if target_roi != roi
        }
        for roi in connectivity_rois
    }

    # Load equivalent models
    LOGGER.info("Loading equivalent models...")
    equivalent_models_df = pkl.load(
        open(analysis_results_path, "rb")
    )

    unique_model_layers = list(set(zip(
        equivalent_models_df.query(f"metric == '{metric}'")['model'],
        equivalent_models_df.query(f"metric == '{metric}'")['layer'], strict=False
    )))
    unique_model_layers.sort()

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

    roi_model_layer_dict: dict[str, list[str]] = defaultdict(list)
    for model_layer, _rois in model_layer_roi_dict.items():
        for r in _rois:
            roi_model_layer_dict[r].append(model_layer)
    roi_model_layer_dict = dict(roi_model_layer_dict)

    # Get subjects
    subjects = list(pw_brain_brain_alignment_patterns.subject_predictor_mapping.keys())

    # Compute alignment pattern similarities with connectivity
    LOGGER.info("Computing alignment pattern similarities with connectivity...")

    # Brain-connectivity APS
    brain_connectivity_aps = pw_brain_brain_alignment_patterns.get_alignment_pattern_similarity(
        similarity_metric=similarity_metric,
        on="connectivity",
        connectivity_reference_alignment_patterns=connectivity_based_ap,
        valid_rois=list(connectivity_based_ap.keys())
    )

    # Trained model-connectivity APS
    trained_model_connectivity_aps = trained_model_brain_alignment_patterns.get_alignment_pattern_similarity(
        similarity_metric=similarity_metric,
        roi_model_layer_dict=roi_model_layer_dict,
        subjects=subjects,
        reference_alignment_patterns=None,
        connectivity_reference_alignment_patterns=connectivity_based_ap,
        valid_rois=list(connectivity_based_ap.keys())
    )

    # Load or create model colors
    if model_colors_dict_path is not None and model_colors_dict_path.exists():
        LOGGER.info(f"Loading model colors from {model_colors_dict_path}")
        with open(model_colors_dict_path, 'rb') as f:
            model_colors_dict = pkl.load(f)
    else:
        LOGGER.info("Generating model colors...")
        cmap = plt.get_cmap("tab20", len(unique_model_layers_names))
        model_colors_dict = {model_layer: cmap(i) for i, model_layer in enumerate(unique_model_layers_names)}
        if model_colors_dict_path is not None:
            LOGGER.info(f"Saving model colors to {model_colors_dict_path}")
            with open(model_colors_dict_path, 'wb') as f:
                pkl.dump(model_colors_dict, f)

    # Create overview plot
    if save_plots:
        LOGGER.info("Creating overview plot...")
        similarities = brain_connectivity_aps

        plot_connectivity_alignment_pattern_overview(
            brain_brain_aps=similarities,
            trained_model_connectivity_aps=trained_model_connectivity_aps,
            connectivity_rois=connectivity_rois,
            model_colors_dict=model_colors_dict,
            plot_path=plot_path,
            metric=metric,
            roi_model_layer_dict=roi_model_layer_dict,
        )

        ##### Plot connectivity-derived alignment pattern for each predictor ROI #####
        LOGGER.info("Plotting connectivity-derived alignment pattern for each predictor ROI...")
        for predictor_roi in connectivity_rois:
            plot_connectivity_alignment_pattern_per_roi(
                predictor_roi=predictor_roi,
                connectivity_rois=connectivity_rois,
                connectivity_based_ap=connectivity_based_ap,
                connectivity_based_ap_std=connectivity_based_ap_std,
                plot_path=plot_path,
                pw_brain_brain_alignment_patterns=pw_brain_brain_alignment_patterns,
                pw_brain_brain_alignment_patterns_other_metric=pw_brain_brain_alignment_patterns_other_metric,
                metric=metric,
                other_metric=other_metric,
            )

    # Random connectivity analysis (if provided)
    if random_connectivity_aps_path is not None and save_plots:
        LOGGER.info("Performing random connectivity analysis...")
        random_connectivity_files = list(random_connectivity_aps_path.glob("connectivity_aps_*.pkl"))
        LOGGER.info(f"Found {len(random_connectivity_files)} random connectivity files")

        # Compute random brain-connectivity APS (per-subject, not just means)
        LOGGER.info("Computing APS for random connectivity matrices...")
        random_brain_connectivity_aps_per_subject = []
        random_brain_connectivity_aps_means = []

        for random_connectivity_aps_file in random_connectivity_files:
            with open(random_connectivity_aps_file, 'rb') as f:
                random_connectivity_aps_meta = pkl.load(f)
                random_connectivity_aps = random_connectivity_aps_meta["connectivity_aps"]

                # Build connectivity-based AP dict for this random connectivity
                random_connectivity_based_ap = {
                    roi: dict(zip([_roi for _roi in connectivity_rois if _roi != roi],
                                  random_connectivity_aps[:, connectivity_rois.index(roi), :].mean(axis=0), strict=False))
                    for roi in connectivity_rois
                }

            tmp = pw_brain_brain_alignment_patterns.get_alignment_pattern_similarity(
                similarity_metric=similarity_metric,
                on="connectivity",
                connectivity_reference_alignment_patterns=random_connectivity_based_ap,
                valid_rois=list(connectivity_based_ap.keys())
            )
            # Store per-subject APS for boxplot
            random_brain_connectivity_aps_per_subject.append(tmp)
            # Store means for CI plot
            random_brain_connectivity_aps_means.append({roi: np.array(tmp[roi]).mean() for roi in connectivity_rois})

        # Create plot directory
        random_plot_path = plot_path / "random"
        random_plot_path.mkdir(parents=True, exist_ok=True)

        # Create boxplot comparing true vs random
        # LOGGER.info("Creating boxplot comparison of true vs random connectivity...")
        # x_pos = 0
        # xtick_positions = []
        # fig, axes = plt.subplots(
        #     2, 1,
        #     gridspec_kw={'height_ratios': [1, 0.5], "hspace": 0.5},
        #     figsize=(12, 8)
        # )

        # roi_spacing = 2
        # offset = 0.25

        # flierprops = dict(
        #     marker='o',
        #     markersize=3,
        #     markerfacecolor='black',
        #     markeredgecolor='black',
        #     alpha=0.6
        # )

        # for roi in connectivity_rois:
        #     # True connectivity (observed)
        #     true_scores = similarities[roi]

        #     # Random connectivity (all subjects across all random matrices)
        #     random_scores = []
        #     for random_aps in random_brain_connectivity_aps_per_subject:
        #         random_scores.extend(random_aps[roi])

        #     # Left box: True connectivity
        #     axes[0].boxplot(
        #         true_scores,
        #         positions=[x_pos - offset],
        #         widths=0.30,
        #         patch_artist=True,
        #         boxprops=dict(facecolor='black', edgecolor='black'),
        #         medianprops=dict(color='white'),
        #         whiskerprops=dict(color='black'),
        #         capprops=dict(color='black'),
        #         flierprops=flierprops
        #     )

        #     # Right box: Random connectivity
        #     axes[0].boxplot(
        #         random_scores,
        #         positions=[x_pos + offset],
        #         widths=0.30,
        #         patch_artist=True,
        #         boxprops=dict(facecolor='white', edgecolor='black'),
        #         medianprops=dict(color='black'),
        #         whiskerprops=dict(color='black'),
        #         capprops=dict(color='black'),
        #         flierprops=flierprops
        #     )

        #     # Center tick for this ROI
        #     xtick_positions.append(x_pos)

        #     # Move to next ROI
        #     x_pos += roi_spacing

        # # Legend
        # observed_patch = Patch(
        #     facecolor='black',
        #     edgecolor='black',
        #     label='Connectivity (observed)'
        # )

        # random_patch = Patch(
        #     facecolor='white',
        #     edgecolor='black',
        #     label='Connectivity (random)'
        # )

        # # Axis styling
        # axes[0].set_xticks(xtick_positions)
        # axes[0].set_xticklabels(connectivity_rois, rotation=90)

        # axes[1].legend(
        #     handles=[observed_patch, random_patch],
        #     loc='center',
        #     fontsize=9,
        #     title='Legend',
        #     title_fontsize=11,
        #     frameon=True,
        #     fancybox=True,
        #     shadow=True
        # )

        # axes[0].axhline(0, color='k', linestyle='--', alpha=0.2)
        # axes[1].axis('off')
        # sns.despine()

        # axes[0].set_ylabel("Alignment pattern similarity \n (Pearson R)")
        # axes[0].set_xlabel("ROIs")
        # axes[0].set_title("Alignment pattern similarity to connectivity")

        # save_this(random_plot_path, fname='random_connectivity_aps_boxplot', transparent=False)
        # plt.close()

        # Also create the mean + CI plot (original version)
        similarities = brain_connectivity_aps
        plot_random_connectivity_comparison(
            brain_brain_aps=similarities,
            random_brain_connectivity_aps_means=random_brain_connectivity_aps_means,
            connectivity_rois=connectivity_rois,
            plot_path=random_plot_path,
            metric=metric,
        )

    LOGGER.info("Analysis complete!")


if __name__ == '__main__':
    main()

