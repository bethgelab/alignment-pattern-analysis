"""
Plotting functions for alignment pattern similarity analyses.

This module contains plotting functions extracted from analysis scripts
for creating alignment pattern similarity visualizations.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from multitasking.plot_creation.plot_utils import save_this

LOGGER = logging.getLogger(__name__)


def plot_alignment_pattern_overview_boxplot(
    brain_brain_aps: dict,
    model_brain_aps: dict,
    roi_list: list,
    model_colors_dict: dict,
    plot_path: Path,
    metric: str,
    roi_model_layer_dict: Optional[dict] = None,
    roi_spacing: float = 2,
    figsize: tuple = (12, 8),
    formats: Optional[list] = None,
):
    """
    Create an overview boxplot showing alignment pattern brain_brain_aps.
    
    Parameters
    ----------
    brain_brain_aps : dict
        Dictionary mapping ROI to list of similarity scores (brain-brain).
    model_brain_aps : dict
        Dictionary mapping ROI to dict of model_layer -> list of scores.
    roi_list : list
        List of ROI names.
    model_colors_dict : dict
        Dictionary mapping model_layer names to colors.
    plot_path : Path
        Path to save the plot.
    metric : str
        Metric name (e.g., 'rsa', 'linear_predictivity').
    roi_model_layer_dict : dict, optional
        Dictionary mapping ROI to list of model_layer names. If None, uses all models for each ROI.
    roi_spacing : float
        Spacing between ROI groups.
    figsize : tuple
        Figure size.
    formats : list, optional
        List of formats to save (default: ['svg', 'png', 'pdf']).
    """
    if formats is None:
        formats = ['svg', 'png', 'pdf']
    
    x_pos = 0
    xtick_positions = []
    fig, axes = plt.subplots(
        2, 1,
        gridspec_kw={'height_ratios': [1, 0.5], "hspace": 0.8},
        figsize=figsize
    )
    
    flierprops_roi = dict(
        marker='o',
        markersize=3,
        markerfacecolor='black',
        markeredgecolor='black',
        alpha=0.6
    )
    
    for roi in roi_list:
        # Within-ROI boxplot
        within_scores = brain_brain_aps[roi]
        
        axes[0].boxplot(
            within_scores,
            positions=[x_pos],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor='black', edgecolor='black'),
            medianprops=dict(color='white'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=flierprops_roi
        )
        
        x_pos += 1  # move to model layers for this ROI
        
        # Model layers boxplots
        if roi_model_layer_dict is not None:
            model_list = roi_model_layer_dict.get(roi, list(model_brain_aps[roi].keys()))
        else:
            model_list = list(model_brain_aps[roi].keys())
        
        n_models = len(model_list)
        x_positions = np.linspace(x_pos, x_pos + n_models - 1, n_models)
        
        sorted_models = sorted(
            [(k, v) for k, v in model_brain_aps[roi].items() if k in model_list],
            key=lambda kv: np.mean(kv[1]),
            reverse=True  # highest → lowest
        )
        
        for j, (model_layer, aps_scores) in enumerate(sorted_models):
            flierprops_model = dict(
                marker='^',
                markersize=2,
                markerfacecolor=model_colors_dict[model_layer],
                markeredgecolor=model_colors_dict[model_layer],
                alpha=0.7
            )
            axes[0].boxplot(
                aps_scores,
                positions=[x_positions[j]],
                widths=0.6,
                patch_artist=True,
                boxprops=dict(facecolor=model_colors_dict[model_layer],
                              edgecolor=model_colors_dict[model_layer]),
                medianprops=dict(color='black'),
                whiskerprops=dict(color=model_colors_dict[model_layer]),
                capprops=dict(color=model_colors_dict[model_layer]),
                flierprops=flierprops_model
            )
        
        # Shading behind this ROI group
        # axes[0].fill_betweenx(
        #     [-0.75, 1],
        #     x_pos - 1,
        #     x_positions[-1] + 1,
        #     color='grey',
        #     alpha=0.1
        # )
        
        # Tick at ROI center
        xtick_positions.append((x_pos - 1 + x_positions[-1]) / 2)
        
        # Update x position for next ROI
        x_pos = x_positions[-1] + roi_spacing
    
    # Axis labels
    axes[0].set_xticks(xtick_positions)
    axes[0].set_xticklabels(roi_list, rotation=90)
    
    # Legend Panel
    legend_elements = [
        Patch(facecolor=color, label=model_layer[:40])
        for model_layer, color in model_colors_dict.items()
    ]
    
    legend_elements.insert(0,
                           Patch(facecolor='k',
                                 label=f"brain activity"))
    
    axes[1].legend(
        handles=legend_elements,
        loc='center',
        fontsize=9,
        title='Predictor feature spaces',
        title_fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=False
    )
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.2, zorder=0)
    axes[1].axis('off')
    sns.despine()
    axes[0].set_ylabel("Alignment pattern similarity \n (Pearson R)")
    axes[0].set_xlabel("ROIs")
    
    save_this(plot_path, fname=f"alignment_pattern_overview_{metric}", formats=formats,
              transparent=False)
    plt.close()


def plot_alignment_pattern_overview_scatter(
    brain_brain_aps: dict,
    model_brain_aps: dict,
    roi_list: list,
    model_colors_dict: dict,
    plot_path: Path,
    metric: str,
    roi_model_layer_dict: Optional[dict] = None,
    roi_spacing: float = 2,
    figsize: tuple = (12, 8),
    formats: Optional[list] = None,
):
    """
    Create an overview scatterplot with error bars showing alignment pattern brain_brain_aps.
    
    Parameters
    ----------
    brain_brain_aps : dict
        Dictionary mapping ROI to list of similarity scores (brain-brain).
    model_brain_aps : dict
        Dictionary mapping ROI to dict of model_layer -> list of scores.
    roi_list : list
        List of ROI names.
    model_colors_dict : dict
        Dictionary mapping model_layer names to colors.
    plot_path : Path
        Path to save the plot.
    metric : str
        Metric name (e.g., 'rsa', 'linear_predictivity').
    roi_model_layer_dict : dict, optional
        Dictionary mapping ROI to list of model_layer names. If None, uses all models for each ROI.
    roi_spacing : float
        Spacing between ROI groups.
    figsize : tuple
        Figure size.
    formats : list, optional
        List of formats to save (default: ['svg', 'png', 'pdf']).
    """
    if formats is None:
        formats = ['svg', 'png', 'pdf']
    
    x_pos = 0
    xtick_positions = []
    fig, axes = plt.subplots(
        2, 1,
        gridspec_kw={'height_ratios': [1, 0.5], "hspace": 0.8},
        figsize=figsize
    )
    
    for roi in roi_list:
        # --- Within-ROI: mean ± 95% CI ---
        within_scores = np.array(brain_brain_aps[roi])
        mean_w = np.mean(within_scores)
        sem_w = np.std(within_scores, ddof=1) / np.sqrt(len(within_scores))
        ci95_w = 1.96 * sem_w
        
        axes[0].errorbar(
            x_pos, mean_w, yerr=ci95_w,
            fmt='o', markersize=3, color='black',
            ecolor='black', elinewidth=1.5, capsize=3
        )
        
        # store for xticks
        x_pos_roistart = x_pos
        
        x_pos += 1
        
        # --- Model layers: mean ± 95% CI for each layer ---
        if roi_model_layer_dict is not None:
            model_list = roi_model_layer_dict.get(roi, list(model_brain_aps[roi].keys()))
        else:
            model_list = list(model_brain_aps[roi].keys())
        
        n_models = len(model_list)
        x_positions = np.linspace(x_pos, x_pos + n_models - 1, n_models)
        
        sorted_models = sorted(
            [(k, v) for k, v in model_brain_aps[roi].items() if k in model_list],
            key=lambda kv: np.mean(kv[1]),
            reverse=True
        )
        
        for j, (model_layer, aps_scores) in enumerate(sorted_models):
            aps_scores = np.array(aps_scores)
            mean_m = np.mean(aps_scores)
            sem_m = np.std(aps_scores, ddof=1) / np.sqrt(len(aps_scores))
            ci95_m = 1.96 * sem_m
            
            jitter = np.random.uniform(-0.1, 0.1, size=len(aps_scores))
            axes[0].scatter(
                np.ones_like(aps_scores) * x_positions[j] + jitter,
                aps_scores,
                color=model_colors_dict[model_layer],
                alpha=0.3,
                edgecolor='none',
                s=2,
                marker='.'
            )
            
            axes[0].errorbar(
                x_positions[j], mean_m, yerr=ci95_m,
                fmt='o', markersize=3,
                color=model_colors_dict[model_layer],
                ecolor=model_colors_dict[model_layer],
                elinewidth=1.2, capsize=3
            )
        
        # Shading behind this ROI group
        axes[0].fill_betweenx(
            [0, 0.03],
            x_pos_roistart-0.7,
            x_positions[-1]+0.7,
            color='grey',
            alpha=0.1,
            zorder=0,
            edgecolor='none'
        )
        
        # ROI tick at center
        xtick_positions.append((x_pos_roistart + x_positions[-1]) / 2)
        
        # Update for next ROI
        x_pos = x_positions[-1] + roi_spacing
    
    # Axis labels
    axes[0].set_xticks(xtick_positions)
    axes[0].set_xticklabels(roi_list, rotation=90)
    
    # Legend Panel
    legend_elements = [
        Patch(facecolor=color, label=model_layer[:30])
        for model_layer, color in model_colors_dict.items()
    ]
    
    legend_elements.insert(0,
                           Patch(facecolor='k',
                                 label=f"brain activity"))
    
    axes[1].legend(
        handles=legend_elements,
        loc='center',
        fontsize=9,
        title='Predictor feature spaces',
        title_fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=False
    )
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.2, zorder=0)
    axes[1].axis('off')
    sns.despine()
    axes[0].set_ylabel("Alignment pattern similarity \n (Pearson R)")
    axes[0].set_xlabel("ROIs")
    
    save_this(plot_path, fname=f"alignment_pattern_overview_SEM_{metric}", formats=formats,
              transparent=False)
    plt.close()


def plot_alignment_pattern_per_roi(
    roi: str,
    roi_list: list,
    brain_brain_aps: dict,
    model_brain_aps: dict,
    model_colors_dict: dict,
    plot_path: Path,
    metric: str,
    pw_brain_brain_alignment_patterns,
    model_brain_alignment_patterns,
    roi_model_layer_dict: dict,
    figsize: tuple = (8, 2),
    formats: Optional[list] = None,
):
    """
    Create a per-ROI alignment pattern plot with alignment patterns and scatter.
    
    Parameters
    ----------
    roi : str
        The ROI to plot.
    roi_list : list
        List of all ROI names.
    brain_brain_aps : dict
        Dictionary mapping ROI to list of similarity scores.
    model_brain_aps : dict
        Dictionary mapping ROI to dict of model_layer -> list of scores.
    model_colors_dict : dict
        Dictionary mapping model_layer names to colors.
    plot_path : Path
        Path to save the plot.
    metric : str
        Metric name.
    pw_brain_brain_alignment_patterns
        PairwiseBrainBrainAlignmentPatterns object.
    model_brain_alignment_patterns
        ModelBrainAlignmentPatterns object.
    roi_model_layer_dict : dict
        Dictionary mapping ROI to list of model_layer names.
    figsize : tuple
        Figure size.
    formats : list, optional
        List of formats to save (default: ['svg', 'png', 'pdf']).
    """
    if formats is None:
        formats = ['svg', 'png', 'pdf']
    
    fig, axes = plt.subplots(
        1, 2,
        figsize=figsize,
        gridspec_kw={'width_ratios': [2, 1], "wspace": 0.3}
    )
    
    # Left panel: shading
    # Brain–brain alignment patterns
    brain_ap_matrix = pw_brain_brain_alignment_patterns.alignment_pattern_dict_by_roi[roi]
    brain_mean = brain_ap_matrix.mean(axis=1)
    brain_std = brain_ap_matrix.std(axis=1)
    
    axes[0].fill_between(range(len(roi_list)), brain_mean - brain_std, brain_mean + brain_std,
                         color='k', alpha=0.1, edgecolor='none')
    axes[0].plot(range(len(roi_list)), brain_mean, color='k', linewidth=2)
    
    # Models
    for model_layer in roi_model_layer_dict[roi]:
        # Get pattern matrix: shape (n_rois, n_subjects)
        # For ModelBrainAlignmentPatterns, predictor_subject is "" and predictor_roi is "model__layer"
        model_mat = model_brain_alignment_patterns.alignment_pattern_data.get_pattern_matrix(
            model_layer, roi_list
        )
        model_mean = model_mat.mean(axis=1)
        model_std = model_mat.std(axis=1)
        color = model_colors_dict[model_layer]
        
        # axes[0].fill_between(range(len(roi_list)), model_mean - model_std, model_mean + model_std,
        #                      color=color, alpha=0.15, edgecolor='none')
        axes[0].plot(range(len(roi_list)), model_mean, color=color, linewidth=2)
    
    axes[0].set_xticks(range(len(roi_list)))
    axes[0].set_xticklabels(roi_list, rotation=90)
    axes[0].set_xlabel("Target ROIs")
    axes[0].set_ylabel(f"Alignment score ({metric})")
    axes[0].set_title(f"{roi}: Mean ± std alignment patterns")
    
    # Right panel: scatter for this ROI
    x_pos = 0
    scatter_width = 0.15  # jitter width
    x_labels = []
    
    # Brain
    brain_vals = brain_brain_aps[roi]
    jitter = np.random.uniform(-scatter_width, scatter_width, size=len(brain_vals))
    axes[1].scatter(
        np.ones_like(brain_vals) * x_pos + jitter,
        brain_vals,
        color='k',
        alpha=0.6,
    )
    x_labels.append(roi)
    x_pos += 1
    
    # Models
    sorted_models = sorted(
        model_brain_aps[roi].items(),
        key=lambda kv: np.mean(kv[1]),
        reverse=True  # highest → lowest
    )
    
    for model_layer, model_vals in sorted_models:
        jitter = np.random.uniform(-scatter_width, scatter_width, size=len(model_vals))
        axes[1].scatter(
            np.ones_like(model_vals) * x_pos + jitter,
            model_vals,
            color=model_colors_dict[model_layer],
            alpha=0.6,
        )
        x_labels.append(model_layer[:20])  # truncate for readability
        x_pos += 1
    
    axes[1].set_xlabel("Predictor feature spaces")
    axes[1].set_xticks(range(x_pos))
    axes[1].set_xticklabels(x_labels, rotation=90)
    axes[1].set_ylabel("Alignment similarity (Pearson R)")
    axes[1].set_title(f"Alignment Pattern brain_brain_aps")
    
    sns.despine()
    save_this(plot_path, fname=f"{metric}_ap_{roi}", transparent=False, formats=formats)
    plt.close()


def plot_connectivity_alignment_pattern_overview(
    brain_brain_aps: dict,
    trained_model_connectivity_aps: dict,
    connectivity_rois: list,
    model_colors_dict: dict,
    plot_path: Path,
    metric: str,
    roi_model_layer_dict: dict,
    roi_spacing: float = 2,
    offset: float = 0.18,
    figsize: tuple = (12, 8),
    formats: Optional[list] = None,
):
    """
    Create an overview boxplot for connectivity-based alignment patterns.
    
    Parameters
    ----------
    brain_brain_aps : dict
        Dictionary mapping ROI to list of similarity scores (brain-connectivity).
    trained_model_connectivity_aps : dict
        Dictionary mapping ROI to dict of model_layer -> list of scores.
    connectivity_rois : list
        List of ROI names.
    model_colors_dict : dict
        Dictionary mapping model_layer names to colors.
    plot_path : Path
        Path to save the plot.
    metric : str
        Metric name.
    roi_model_layer_dict : dict
        Dictionary mapping ROI to list of model_layer names.
    roi_spacing : float
        Spacing between ROI groups.
    offset : float
        Horizontal offset for model boxes.
    figsize : tuple
        Figure size.
    formats : list, optional
        List of formats to save (default: ['svg', 'png', 'pdf']).
    """
    if formats is None:
        formats = ['svg', 'png', 'pdf']
    
    x_pos = 0
    xtick_positions = []
    fig, axes = plt.subplots(
        2, 1,
        gridspec_kw={'height_ratios': [1, 0.5], "hspace": 0.7},
        figsize=figsize
    )
    
    flierprops_roi = dict(
        marker='o',
        markersize=3,
        markerfacecolor='black',
        markeredgecolor='black',
        alpha=0.6
    )
    
    for roi in connectivity_rois:
        within_scores = brain_brain_aps[roi]
        
        # Within-ROI boxplot
        axes[0].boxplot(
            within_scores,
            positions=[x_pos],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor='black', edgecolor='black'),
            medianprops=dict(color='white'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=flierprops_roi
        )
        
        x_pos += 1
        
        # Models for this ROI
        model_list = roi_model_layer_dict[roi]
        n_models = len(model_list)
        x_centers = np.linspace(x_pos, x_pos + n_models - 1, n_models)
        
        sorted_models = sorted(
            [(k, v) for k, v in trained_model_connectivity_aps[roi].items() if k in model_list],
            key=lambda kv: np.mean(kv[1]),
            reverse=True
        )
        
        for j, (model_layer, trained_scores) in enumerate(sorted_models):
            color = model_colors_dict[model_layer]
            
            flierprops_model = dict(
                marker='^',
                markersize=2,
                markerfacecolor=color,
                markeredgecolor=color,
                alpha=0.7
            )
            
            axes[0].boxplot(
                trained_scores,
                positions=[x_centers[j] + offset],
                widths=0.35,
                patch_artist=True,
                boxprops=dict(facecolor=color, edgecolor=color),
                medianprops=dict(color='black'),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
                flierprops=flierprops_model
            )
        
        # ROI shading
        axes[0].fill_betweenx(
            [-0.6, 1],
            x_pos - 1,
            x_centers[-1] + 1,
            color='grey',
            alpha=0.1
        )
        
        xtick_positions.append((x_pos - 1 + x_centers[-1]) / 2)
        
        x_pos = x_centers[-1] + roi_spacing
    
    # Axis labels
    axes[0].set_xticks(xtick_positions)
    axes[0].set_xticklabels(connectivity_rois, rotation=90)
    
    # Legend
    legend_elements = [
        Patch(facecolor=model_colors_dict[model_layer], label=model_layer[:40])
        for model_layer in model_colors_dict.keys()
    ]
    legend_elements.insert(0, Patch(facecolor='k', label=f"subject-connectivity"))
    
    axes[1].legend(
        handles=legend_elements,
        loc='center',
        fontsize=9,
        title='Model-Layer Combinations',
        title_fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=True
    )
    
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.2)
    axes[1].axis('off')
    sns.despine()
    axes[0].set_ylabel("Alignment pattern similarity \n (Pearson R)")
    axes[0].set_xlabel("ROIs")
    axes[0].set_title("Alignment pattern similarity to connectivity")
    
    save_this(plot_path, fname=f"alignment_pattern_overview_{metric}_connectivity_trained",
              formats=formats, transparent=False)
    plt.close()


def plot_connectivity_alignment_pattern_per_roi(
    predictor_roi: str,
    connectivity_rois: list,
    connectivity_based_ap: dict,
    connectivity_based_ap_std: dict,
    plot_path: Path,
    pw_brain_brain_alignment_patterns,
    pw_brain_brain_alignment_patterns_other_metric,
    metric: str,
    other_metric: str,
    formats: Optional[list] = None,
):
    """
    Plot connectivity-derived alignment pattern for a predictor ROI.
    
    Parameters
    ----------
    predictor_roi : str
        The predictor ROI to plot.
    connectivity_rois : list
        List of all connectivity ROIs.
    connectivity_based_ap : dict
        Dictionary mapping ROI to dict of target_roi -> mean value.
    connectivity_based_ap_std : dict
        Dictionary mapping ROI to dict of target_roi -> std value.
    plot_path : Path
        Path to save the plot.
    pw_brain_brain_alignment_patterns
        PairwiseBrainBrainAlignmentPatterns object for main metric.
    pw_brain_brain_alignment_patterns_other_metric
        PairwiseBrainBrainAlignmentPatterns object for other metric.
    metric : str
        Main metric name.
    other_metric : str
        Other metric name.
    formats : list, optional
        List of formats to save (default: ['svg', 'png', 'pdf']).
    """
    if formats is None:
        formats = ['svg', 'png', 'pdf']
    
    mean_vals = np.array([val for key, val in connectivity_based_ap[predictor_roi].items()])
    std_vals = np.array([val for key, val in connectivity_based_ap_std[predictor_roi].items()])
    plt.figure(figsize=(6, 2))
    plt.plot(range(len(connectivity_rois)-1), 
             mean_vals, color='k')
    plt.fill_between(
        range(len(connectivity_rois)-1),
        mean_vals - std_vals,
        mean_vals + std_vals,
        color='k',
        alpha=0.1
    )
    plt.xticks(range(len(connectivity_rois)-1), 
               [key for key in connectivity_based_ap[predictor_roi].keys()], rotation=90)
    
    valid_indices = [connectivity_rois.index(roi) for roi in connectivity_rois if roi != predictor_roi]
    # Add brain-brain alignment pattern for this predictor ROI
    metric_color_dict = {'rsa': 'darkred', 'linear_predictivity': 'darkblue'}
    brain_ap_matrix = pw_brain_brain_alignment_patterns.alignment_pattern_dict_by_roi[predictor_roi]
    brain_mean = brain_ap_matrix[valid_indices].mean(axis=1)
    brain_std = brain_ap_matrix[valid_indices].std(axis=1)
    LOGGER.info(f"Plotting fMRI-derived brain-brain alignment pattern for {predictor_roi}")
    plt.fill_between(range(len(connectivity_rois)-1), brain_mean - brain_std, brain_mean + brain_std,
                         color=metric_color_dict[metric], alpha=0.1)
    plt.plot(range(len(connectivity_rois)-1), brain_mean, color=metric_color_dict[metric], linewidth=2)
    
    brain_ap_matrix = pw_brain_brain_alignment_patterns_other_metric.alignment_pattern_dict_by_roi[predictor_roi]
    brain_mean = brain_ap_matrix[valid_indices].mean(axis=1)
    brain_std = brain_ap_matrix[valid_indices].std(axis=1)
    LOGGER.info(f"Plotting fMRI-derived brain-brain alignment pattern for {predictor_roi}")
    plt.fill_between(range(len(connectivity_rois)-1), brain_mean - brain_std, brain_mean + brain_std,
                         color=metric_color_dict[other_metric], alpha=0.1)
    plt.plot(range(len(connectivity_rois)-1), brain_mean, color=metric_color_dict[other_metric], linewidth=2)
    
    plt.xlabel("ROIs")
    plt.ylabel("Alignment score \n (RSA/LP/streamline density)")
    plt.title(f"Connectivity-derived alignment pattern \n {predictor_roi}")
    sns.despine()
    save_this(plot_path, fname=f"alignment_pattern_connectivity_{predictor_roi}",
              formats=formats, transparent=False)
    plt.close()


def plot_random_connectivity_comparison(
    brain_brain_aps: dict,
    random_brain_connectivity_aps_means: list,
    connectivity_rois: list,
    plot_path: Path,
    metric: str,
    roi_spacing: float = 2,
    offset: float = 0.25,
    figsize: tuple = (12, 8),
    formats: Optional[list] = None,
):
    """
    Create a plot comparing observed vs random connectivity alignment patterns.
    
    Parameters
    ----------
    brain_brain_aps : dict
        Dictionary mapping ROI to list of similarity scores (observed).
    random_brain_connectivity_aps_means : list
        List of dicts, each mapping ROI to mean score for a random connectivity.
    connectivity_rois : list
        List of ROI names.
    plot_path : Path
        Path to save the plot.
    metric : str
        Metric name.
    roi_spacing : float
        Spacing between ROI groups.
    offset : float
        Horizontal offset for observed vs random.
    figsize : tuple
        Figure size.
    formats : list, optional
        List of formats to save (default: ['svg', 'png', 'pdf']).
    """
    if formats is None:
        formats = ['svg', 'png', 'pdf']
    
    x_pos: float = 0.0
    xtick_positions = []
    fig, axes = plt.subplots(
        2, 1,
        gridspec_kw={'height_ratios': [1, 0.5], "hspace": 0.5},
        figsize=figsize
    )
    
    for roi in connectivity_rois:
        # Observed connectivity
        within_score_mean = np.array(brain_brain_aps[roi]).mean()
        
        # Random connectivity distribution (means)
        within_scores_random = [el[roi] for el in random_brain_connectivity_aps_means]
        
        lower_ci_bound = np.percentile(within_scores_random, 2.5)
        upper_ci_bound = np.percentile(within_scores_random, 97.5)
        
        color = 'r' if within_score_mean > upper_ci_bound else 'k'
        
        # Observed scatter
        axes[0].scatter(x_pos - offset, within_score_mean, color=color)
        
        # Random CI
        axes[0].plot(
            [x_pos + offset] * 2,
            [lower_ci_bound, upper_ci_bound],
            color='k'
        )
        
        xtick_positions.append(x_pos)
        
        x_pos += roi_spacing
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8,
               label='avg. brain-connectivity APS score (outside CI)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8,
               label='avg. brain-connectivity APS score (within CI)'),
        Line2D([0], [0], color='k', linewidth=2,
               label='95% CI of avg. brain-connectivity APS\nto random connectivity graphs')
    ]
    
    axes[0].legend(handles=legend_elements, loc='upper right', frameon=False)
    
    # Axis styling
    axes[0].set_xticks(xtick_positions)
    axes[0].set_xticklabels(connectivity_rois, rotation=90)
    
    axes[1].legend(
        handles=legend_elements,
        loc='center',
        fontsize=9,
        title='Legend',
        title_fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=True
    )
    
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.2)
    axes[1].axis('off')
    sns.despine()
    
    axes[0].set_ylabel("Alignment pattern similarity \n (Pearson R)")
    axes[0].set_xlabel("ROIs")
    axes[0].set_title("Alignment pattern similarity to connectivity")
    
    save_this(plot_path, fname=f'random_connectivity_aps_mean_{metric}', transparent=False, formats=formats)
    plt.close()


def plot_trained_untrained_comparison(
    brain_brain_aps: dict,
    trained_model_brain_aps: dict,
    model_brain_aps: dict,
    roi_list: list,
    model_colors_dict: dict,
    plot_path: Path,
    metric: str,
    roi_model_layer_dict: dict,
    roi_spacing: float = 2,
    offset: float = 0.18,
    figsize: tuple = (12, 8),
    formats: Optional[list] = None,
):
    """
    Create an overview plot comparing trained vs untrained models side by side.
    
    Parameters
    ----------
    brain_brain_aps : dict
        Dictionary mapping ROI to list of similarity scores (brain-brain).
    trained_model_brain_aps : dict
        Dictionary mapping ROI to dict of model_layer -> list of scores (trained).
    model_brain_aps : dict
        Dictionary mapping ROI to dict of model_layer -> list of scores (untrained).
    roi_list : list
        List of ROI names.
    model_colors_dict : dict
        Dictionary mapping model_layer names to colors.
    plot_path : Path
        Path to save the plot.
    metric : str
        Metric name.
    roi_model_layer_dict : dict
        Dictionary mapping ROI to list of model_layer names.
    roi_spacing : float
        Spacing between ROI groups.
    offset : float
        Horizontal offset for trained vs untrained.
    figsize : tuple
        Figure size.
    formats : list, optional
        List of formats to save (default: ['svg', 'png', 'pdf']).
    """
    if formats is None:
        formats = ['svg', 'png', 'pdf']
    
    x_pos = 0
    xtick_positions = []
    fig, axes = plt.subplots(
        2, 1,
        gridspec_kw={'height_ratios': [1, 0.5], "hspace": 0.6},
        figsize=figsize
    )
    
    flierprops_roi = dict(
        marker='o',
        markersize=3,
        markerfacecolor='black',
        markeredgecolor='black',
        alpha=0.6
    )
    
    tick_labels = []
    for i, roi in enumerate(roi_list):
        within_scores = brain_brain_aps[roi]
        
        # --- Within-ROI boxplot ---
        axes[0].boxplot(
            within_scores,
            positions=[x_pos],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor='black', edgecolor='black'),
            medianprops=dict(color='white'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=flierprops_roi
        )
        
        x_pos += 1
        
        # --- Models for this ROI ---
        model_list = roi_model_layer_dict[roi]
        n_models = len(model_list)
        if n_models == 0:
            continue
        tick_labels.append(roi)
        x_centers = np.linspace(x_pos, x_pos + n_models - 1, n_models)
        
        sorted_models = sorted(
            [(k, v) for k, v in trained_model_brain_aps[roi].items() if k in model_list],
            key=lambda kv: np.mean(kv[1]),
            reverse=True
        )
        
        for j, (model_layer, trained_scores) in enumerate(sorted_models):
            untrained_scores = model_brain_aps[roi][model_layer]
            color = model_colors_dict[model_layer]
            
            flierprops_model = dict(
                marker='^',
                markersize=2,
                markerfacecolor=color,
                markeredgecolor=color,
                alpha=0.7
            )
            
            # ---- Untrained on LEFT ----
            axes[0].boxplot(
                untrained_scores,
                positions=[x_centers[j] - offset],
                widths=0.35,
                patch_artist=True,
                boxprops=dict(facecolor='white', edgecolor=color),
                medianprops=dict(color='black'),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
                flierprops=flierprops_model
            )
            
            # ---- Trained on RIGHT ----
            axes[0].boxplot(
                trained_scores,
                positions=[x_centers[j] + offset],
                widths=0.35,
                patch_artist=True,
                boxprops=dict(facecolor=color, edgecolor=color),
                medianprops=dict(color='black'),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
                flierprops=flierprops_model
            )
        
        # ROI shading
        axes[0].fill_betweenx(
            [-0.6, 1],
            x_pos - 1,
            x_centers[-1] + 1,
            color='grey',
            alpha=0.1
        )
        
        xtick_positions.append((x_pos - 1 + x_centers[-1]) / 2)
        x_pos = x_centers[-1] + roi_spacing
    
    # Create legend entries indicating trained vs untrained
    trained_patch = Patch(
        facecolor='grey',
        edgecolor='grey',
        label='Trained'
    )
    
    untrained_patch = Patch(
        facecolor='white',
        edgecolor='black',
        label='Untrained'
    )
    
    # Axis labels and legend
    axes[0].set_xticks(xtick_positions)
    axes[0].set_xticklabels(tick_labels, rotation=90)
    
    legend_elements = [
        Patch(facecolor=model_colors_dict[model_layer], 
              label=model_layer[:40])
        for model_layer in model_colors_dict.keys()
    ]
    legend_elements.insert(0, 
           Patch(facecolor='k', 
          label=f"brain activity"))
    legend_elements.insert(1, trained_patch)
    legend_elements.insert(2, untrained_patch)
    
    axes[1].legend(
        handles=legend_elements,
        loc='center',
        fontsize=9,
        title='Predictor feature spaces',
        title_fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=False
    )
    
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.2)
    axes[1].axis('off')
    sns.despine()
    axes[0].set_ylabel("Alignment pattern similarity \n (Pearson R)")
    axes[0].set_xlabel("ROIs")
    
    save_this(plot_path, fname=f"alignment_pattern_overview_{metric}_trained_untrained_side_side", 
              formats=formats, transparent=False)
    plt.close()


def plot_alignment_patterns_per_roi_simple(
    roi: str,
    roi_list: list,
    brain_brain_aps: dict,
    model_brain_aps: dict,
    model_colors_dict: dict,
    plot_path: Path,
    metric: str,
    pw_brain_brain_alignment_patterns,
    model_brain_alignment_patterns,
    roi_model_layer_dict: dict,
    figsize: tuple = (6, 4),
    formats: Optional[list] = None,
):
    """
    Create a simple per-ROI alignment pattern plot showing alignment patterns only.
    
    Parameters
    ----------
    roi : str
        The ROI to plot.
    roi_list : list
        List of all ROI names.
    brain_brain_aps : dict
        Dictionary mapping ROI to list of similarity scores.
    model_brain_aps : dict
        Dictionary mapping ROI to dict of model_layer -> list of scores.
    model_colors_dict : dict
        Dictionary mapping model_layer names to colors.
    plot_path : Path
        Path to save the plot.
    metric : str
        Metric name.
    pw_brain_brain_alignment_patterns
        PairwiseBrainBrainAlignmentPatterns object.
    model_brain_alignment_patterns
        ModelBrainAlignmentPatterns object.
    roi_model_layer_dict : dict
        Dictionary mapping ROI to list of model_layer names.
    figsize : tuple
        Figure size.
    formats : list, optional
        List of formats to save (default: ['svg', 'png', 'pdf']).
    """
    if formats is None:
        formats = ['svg', 'png', 'pdf']
    
    plt.figure(figsize=figsize)
    
    # Brain–Brain alignment patterns (all subjects)
    brain_ap_matrix = pw_brain_brain_alignment_patterns.alignment_pattern_dict_by_roi[roi]
    brain_mean = brain_ap_matrix.mean(axis=1)
    brain_std = brain_ap_matrix.std(axis=1)
    
    brain_avg_aps = np.average(brain_brain_aps[roi])
    brain_std_aps = np.std(brain_brain_aps[roi])
    
    plt.fill_between(
        range(len(roi_list)),
        brain_mean - brain_std,
        brain_mean + brain_std,
        color='k',
        alpha=0.1,
    )
    
    plt.plot(
        range(len(roi_list)),
        brain_mean,
        color='k',
        linewidth=2,
        label=f"{roi}, APS: {brain_avg_aps:.2f}+-{brain_std_aps:.2f}"
    )
    
    # Model–Brain alignment patterns
    for model_layer in roi_model_layer_dict[roi]:
        # Get pattern matrix: shape (n_rois, n_subjects)
        # For ModelBrainAlignmentPatterns, predictor_subject is "" and predictor_roi is "model__layer"
        model_mat = model_brain_alignment_patterns.alignment_pattern_data.get_pattern_matrix(
            model_layer, roi_list
        )
        
        model_mean = model_mat.mean(axis=1)
        model_std = model_mat.std(axis=1)
        
        color = model_colors_dict[model_layer]
        
        # Shaded region
        plt.fill_between(
            range(len(roi_list)),
            model_mean - model_std,
            model_mean + model_std,
            color=color,
            alpha=0.15,
        )
        
        # Mean line
        avg_aps = np.average(model_brain_aps[roi][model_layer])
        std_aps = np.std(model_brain_aps[roi][model_layer])
        
        plt.plot(
            range(len(roi_list)),
            model_mean,
            color=color,
            linewidth=2,
            label=f"{model_layer[:30]}, APS: {avg_aps:.2f} +- {std_aps:.2f}"
        )
    
    plt.legend()
    plt.xticks(range(len(roi_list)), roi_list, rotation=90)
    
    sns.despine()
    save_this(plot_path, fname=f"random_{metric}_ap_{roi}", transparent=False, formats=formats)
    plt.close()
