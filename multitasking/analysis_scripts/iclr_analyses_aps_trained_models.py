#!/usr/bin/env python3
"""Script to analyze alignment pattern similarity for trained models.

This script performs alignment pattern similarity analysis comparing brain-brain
and model-brain alignment patterns. It was converted from a Jupyter notebook.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import cast

import click
import matplotlib.pyplot as plt
import numpy as np

from multitasking.analyses.alignment_patterns.brain_brain_alignment_patterns import (
    PairwiseBrainBrainAlignmentPatterns,
)
from multitasking.analyses.alignment_patterns.model_brain_alignment_patterns import (
    ModelBrainAlignmentPatterns,
)
from multitasking.plot_creation.alignment_pattern_plots import (
    plot_alignment_pattern_overview_boxplot,
    plot_alignment_pattern_overview_scatter,
    plot_alignment_pattern_per_roi,
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
    '--metric',
    default='rsa',
    help='Metric to use (default: rsa)',
    type=click.Choice(['rsa', 'linear_predictivity']),
)
@click.option(
    '--rois',
    default='V1,V2,V3,V4,V8,FFC,PIT,V3A,V3B,V6,V6A,V7,IPS1,MST,MT,FST,LO1,LO2,LO3',
    help='Comma-separated list of ROIs to analyze',
    type=str,
)
@click.option(
    '--plot-path',
    required=True,
    help='Path to directory where plots will be saved',
    type=click.Path(path_type=Path),
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
    type=click.Choice(['pearson', 'mse', 'mae', 'rank_correlation', 'cosine', 'variance_explained']),
)
def main(
    split: str,
    metric: str,
    rois: str,
    plot_path: Path,
    save_results: bool,
    save_plots: bool,
    log_level: str,
    similarity_metric: str,
):
    """Run alignment pattern similarity analysis for trained models."""
    # Set up logging
    logging.basicConfig(level=getattr(logging, log_level))

    # Parse ROIs
    roi_list = [r.strip() for r in rois.split(',')]
    LOGGER.info(f"Analyzing ROIs: {roi_list}")

    # Create plot directory
    plot_path.mkdir(parents=True, exist_ok=True)

    SCRATCH = Path(os.environ.get("SCRATCH", "/scratch"))
    # Load intersubject dataframes (up to 2)
    with open(SCRATCH / "equivalent_models_df.pkl", "rb") as f:
        equivalent_models_df = pickle.load(f)
    with open(SCRATCH / "pairwise_subject_df.pkl", "rb") as f:
        pairwise_subject_df = pickle.load(f)
    with open(SCRATCH / "full_df.pkl", "rb") as f:
        full_df = pickle.load(f)

    #########################################################
    # Do alignment pattern similarity analysis
    LOGGER.info("Computing brain-brain alignment patterns...")
    pw_brain_brain_alignment_patterns = PairwiseBrainBrainAlignmentPatterns(
        predictor_type="brain",
        predictor_rois=roi_list,
        target_type="brain",
        target_rois=roi_list,
        metric=metric
    )

    # create an AlignmentPatternData object that stores the alignment patterns as a DataFrame
    pw_brain_brain_alignment_patterns.get_alignment_pattern(pairwise_subject_df, split=split)
    # create a dictionary that stores the alignment patterns (19x50) by ROI
    pw_brain_brain_alignment_patterns.get_alignment_pattern_dict_by_roi(roi_list)

    subjects = list(pw_brain_brain_alignment_patterns.subject_predictor_mapping.keys())

    result = pw_brain_brain_alignment_patterns.get_alignment_pattern_similarity(
        similarity_metric=similarity_metric,
        on="exclude_subject",
        subjects=subjects,
    )
    # Unpack the tuple with proper type annotations
    # When on="exclude_subject", returns:
    # tuple[dict[str, dict[str, list[float]]], dict[str, list[float]], dict[str, dict[str, list[float]]]]
    # We only need the third element (reference_patterns_by_subject)
    _, alignment_pattern_similarity_pw, reference_patterns_by_subject_raw = cast(
        tuple[dict[str, dict[str, list[float]]], dict[str, list[float]], dict[str, dict[str, list[float]]]],
        result
    )
    # Convert list[float] to np.ndarray for ModelBrainAlignmentPatterns
    # which expects dict[str, dict[str, np.ndarray]] when subjects is provided
    reference_patterns_by_subject: dict[str, dict[str, np.ndarray]] = {
        subject: {
            roi: np.array(pattern)
            for roi, pattern in patterns.items()
        }
        for subject, patterns in reference_patterns_by_subject_raw.items()
    }
    #########################################################

    # get all unique model-layer combinations
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
        # get all ROIs for a given model-layer combination
        ml_rois = equivalent_models_df.query(
            f"metric == '{metric}' & model =='{model}' & layer == '{layer}'"
        )["roi"].to_list()
        for r in ml_rois:
            model_layer_roi_dict[f"{model}__{layer}"].append(r)

    # build a dictionary that maps each ROI to all model-layer combinations
    roi_model_layer_dict: dict[str, list[str]] = {}
    for model_layer, _rois in model_layer_roi_dict.items():
        for r in _rois:
            if r not in roi_model_layer_dict:
                roi_model_layer_dict[r] = []
            roi_model_layer_dict[r].append(model_layer)

    # Model-brain alignment patterns
    LOGGER.info("Computing model-brain alignment patterns...")
    model_brain_alignment_patterns = ModelBrainAlignmentPatterns(
        predictor_type="model",
        predictor_names=unique_model_layers_names,
        target_type="brain",
        target_names=roi_list,
        metric=metric
    )

    model_brain_alignment_patterns.get_alignment_pattern(full_df, split=split)

    # When subjects is provided, reference_alignment_patterns should be
    # dict[str, dict[str, np.ndarray]] (subject -> roi -> array)
    model_aps = model_brain_alignment_patterns.get_alignment_pattern_similarity(
        similarity_metric=similarity_metric,
        roi_model_layer_dict=roi_model_layer_dict,
        subjects=subjects,
        reference_alignment_patterns=reference_patterns_by_subject
    )

    # Save results
    if save_results:
        LOGGER.info("Overriding results path to SCRATCH ...")
        results_dir = SCRATCH / "iclr_analyses_aps_trained_models"
        LOGGER.info(f"Saving results to {results_dir}...")
        results_dir.mkdir(parents=True, exist_ok=True)

        with open(results_dir / f'model_brain_aps_{metric}.pkl', 'wb') as f:
            pickle.dump(model_brain_alignment_patterns, f)

        with open(results_dir / f'brain_brain_aps_{metric}.pkl', 'wb') as f:
            pickle.dump(pw_brain_brain_alignment_patterns, f)

        LOGGER.info(f"Results saved to {results_dir}")

    LOGGER.info("Analysis complete!")

    # Create overview plot
    if save_plots:

        # Load or create model colors
        # equivalent_models_df["model_family"] = equivalent_models_df["model"].apply(
        #     lambda x: x.split("/")[0]
        #     )
        # model_families = equivalent_models_df["model_family"].unique()
        # mf_dict = {mf: [] for mf in model_families}
        # for mf in model_families:
        #     mf_dict[mf] = equivalent_models_df[equivalent_models_df["model_family"] == mf]["model_layer"].unique()
        # model_colors_dict_full, group_cmap_dict = \
        #     generate_grouped_cubehelix_colors(mf_dict,
        #                                     #   rot=0.5,
        #                                     base_cmap_name="hsv")
        # model_colors_dict = {
        #     model_layer: model_colors_dict_full[model_layer]
        #     for model_layer in unique_model_layers_names
        # }
        model_colors_dict_full = pickle.load(open("/scratch/model_colors_dict.pkl", "rb"))
        try:
            model_colors_dict = {
                model_layer: model_colors_dict_full[model_layer]
                for model_layer in unique_model_layers_names
            }
        except KeyError:
            LOGGER.error(f"Model colors dictionary does not contain all model-layer combinations: {unique_model_layers_names}")
            cmap = plt.get_cmap("tab20", len(unique_model_layers_names))
            model_colors_dict = {model_layer: cmap(i) for i, model_layer in enumerate(unique_model_layers_names)}

        LOGGER.info("Creating overview plot...")

        # Boxplot overview
        plot_alignment_pattern_overview_boxplot(
            brain_brain_aps=alignment_pattern_similarity_pw,
            model_brain_aps=model_aps,
            roi_list=roi_list,
            model_colors_dict=model_colors_dict,
            plot_path=plot_path,
            metric=metric,
            roi_model_layer_dict=roi_model_layer_dict,
        )

        # Scatterplot overview
        plot_alignment_pattern_overview_scatter(
            brain_brain_aps=alignment_pattern_similarity_pw,
            model_brain_aps=model_aps,
            roi_list=roi_list,
            model_colors_dict=model_colors_dict,
            plot_path=plot_path,
            metric=metric,
            roi_model_layer_dict=roi_model_layer_dict,
        )

    # Plot alignment patterns per ROI
        LOGGER.info("Creating per-ROI alignment pattern plots...")
        for roi in roi_list:
            plot_alignment_pattern_per_roi(
                roi=roi,
                roi_list=roi_list,
                brain_brain_aps=alignment_pattern_similarity_pw,
                model_brain_aps=model_aps,
                model_colors_dict=model_colors_dict,
                plot_path=plot_path,
                metric=metric,
                pw_brain_brain_alignment_patterns=pw_brain_brain_alignment_patterns,
                model_brain_alignment_patterns=model_brain_alignment_patterns,
                roi_model_layer_dict=roi_model_layer_dict,
            )




if __name__ == '__main__':
    main()

