import logging
from pathlib import Path
from typing import Any, Mapping

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from benedict import benedict
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

from multitasking.fmri_data.roi_utils import RoiMasks
from multitasking.fmri_data.voxel_consistency import (
    get_roi_wise_voxel_consistency,
)
from multitasking.utils.scoresheet_loading import load_scoresheets_from_parent_dir

LOGGER = logging.getLogger(__name__)


def _check_intersubject_results(df: pd.DataFrame) -> None:
    rois = df["roi"].unique()
    rois_2 = df["layer"].unique()
    assert set(rois) == set(rois_2), (
        f"Expected the same ROIs in both roi and layer columns, "
        f"found {set(rois) - set(rois_2)} in roi but not in layer, "
        f"and {set(rois_2) - set(rois)} in layer but not in roi"
    )
    models_aka_loom = df["model"].unique()
    assert len(models_aka_loom) == 1, (
        f"Expected a single 'model' in intersubject results, found {models_aka_loom}"
    )
    assert models_aka_loom.item() == "all-other-subjects", (
        f"Expected 'all-other-subjects' as model, found {models_aka_loom}"
    )


def _get_ncsnr_per_roi(config_dir: Path) \
        -> tuple[dict[str, float], Mapping[str, Any], RoiMasks]:
    ncsnr_median_per_roi: dict[str, float] | None = None
    # Find a config.yaml from any child run directory
    config_paths = list((config_dir).glob("*/config.yaml"))
    if len(config_paths) == 0:
        raise ValueError(f"compare_to_ncsnr: No config.yaml found under {config_dir}")
    if True:  # else:
        cfg = benedict.from_yaml(config_paths[0])
        fmri_cfg = cfg["fmri"]
        rois = fmri_cfg["roi_names"]
        subjects = fmri_cfg["sub_id"]
        if subjects == "all":
            subjects = [f"sub-{i:02d}" for i in range(1, 11)]

        # Build ROI masks consistent with how voxel consistency is loaded
        roi_masks = RoiMasks(rois, fmri_cfg.get("roi_groups", {}))
        roi_masks.extract_rois()

        # Compute per-subject, per-ROI median NCSNR, then aggregate across subjects
        per_subject_roi_medians: dict[str, dict[str, float]] = {
            str(s): {} for s in subjects
        }
        for subject_id in subjects:
            ncsnr_per_roi = get_roi_wise_voxel_consistency(
                fmri_cfg, roi_masks, str(subject_id)
            )
            for roi in rois:
                vals = ncsnr_per_roi[roi]
                per_subject_roi_medians[str(subject_id)][roi] = float(
                    np.nanmedian(vals)
                )

        # Aggregate across subjects: mean over subjects for each ROI
        ncsnr_median_per_roi = {
            roi: float(
                np.nanmean([per_subject_roi_medians[str(s)][roi] for s in subjects])
            )
            for roi in rois
        }
    return ncsnr_median_per_roi, fmri_cfg, roi_masks


@click.command()
@click.option(
    "--noise-ceiling-results-dir",
    "noise_ceiling_results_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="The directory containing noise ceiling " + "results to plot.",
)
@click.option(
    "--compare-to-ncsnr",
    "compare_to_ncsnr",
    is_flag=True,
    default=False,
    help="Add another subplot with the median "
    + "noise ceiling signal-to-noise ratio, to show"
    + " how much the noise-ceiling for each metric"
    + " correlates with it.",
)
@click.option(
    "--compare-to-glasser-probability",
    "compare_to_glasser_probability",
    is_flag=True,
    default=False,
    help="Add another subplot with the mean "
    + "glasser probability, to show how much the noise-ceiling "
    + "correlates with it.",
)
@click.option(
    "--plot-path",
    "plot_path",
    type=click.Path(exists=False, file_okay=False, path_type=Path),
    required=False,
    help="The path to save the plot to. Defaults to "
    + "a ./plots subdirectory of the results directory.",
)
@click.option(
    "--split",
    "split",
    type=str,
    default="both",
    help="The split to plot. One of: 'train', 'test' or 'both'.",
)
def plot_noiseceilings(
    noise_ceiling_results_dir: Path,
    compare_to_ncsnr: bool,
    compare_to_glasser_probability: bool,
    plot_path: Path,
    split: str,
):
    """Plot noise ceilings for all metrics contained in the results folder.

    Find all results files in the given directory.
    Find the same-ROI results and plot a line for each,
         all within the same subplot.
    Create one subplot per metric.

    """
    if plot_path is None:
        plot_path = noise_ceiling_results_dir / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)

    # ql = 0
    # qu = 100

    metric_dfs: dict[str, pd.DataFrame] = {}
    for metric in ["rsa", "procrustes", "versa", "linear_predictivity"]:
        intersubject_df = load_scoresheets_from_parent_dir(
            noise_ceiling_results_dir, filename_metric=metric, old_format=False
        )
        metric_dfs[metric] = intersubject_df

    df = pd.concat(metric_dfs.values())
    if len(df) == 0:
        raise ValueError(f"No results found in {noise_ceiling_results_dir.absolute()}")

    _check_intersubject_results(df)

    subjects = df["subject"].unique()
    metrics = df["metric"].unique()
    rois = df["roi"].unique()

    # rainbow colors
    colors = {"linear_predictivity": "darkblue", "rsa": "darkred"}
    # colors = sns.color_palette("rainbow", len(metrics))
    colors_rois = sns.color_palette("rainbow", len(rois))

    if split == "both":
        splits = ["train", "test"]
    else:
        splits = [split]

    # If requested, load NCSNR per subject/ROI from fmri config + dataset files
    if compare_to_ncsnr:
        # Find a config.yaml from any child run directory
        # config_paths = list((noise_ceiling_results_dir).glob("*/config.yaml"))
        # if len(config_paths) == 0:
        #     raise ValueError(
        #         f"compare_to_ncsnr: No config.yaml found under
        #       {noise_ceiling_results_dir}"
        #     )
        ncsnr_median_per_roi, fmri_cfg, roi_masks = _get_ncsnr_per_roi(
            noise_ceiling_results_dir
        )

    for split in splits:
        cmp_to_str = ""
        if compare_to_ncsnr:
            cmp_to_str = "_ncsnr"
        outfilename = f"noise_ceilings_per_roi_{split}{cmp_to_str}.png"

        df_split = df[df["split"] == split]

        n_rows = 1 # len(metrics)
        if compare_to_ncsnr:
            n_rows += 1
        fig_height = 6 + (6 if compare_to_ncsnr else 0)
        fig, axes = plt.subplots(nrows=n_rows, ncols=1, figsize=(10, fig_height))
        if n_rows == 1:
            axes = [axes]
        for metric_idx, metric in enumerate(metrics):
            ax = axes[0]
            df_metric = df_split[df_split["metric"] == metric]

            # reduce to noise ceilings only:
            df_metric = df_metric[df_metric["layer"] == df_metric["roi"]]
            df_metric = df_metric.drop_duplicates(
                subset=["subject", "score", "layer", "roi"]
            )

            metric_name = metric.replace("_", " ").title()

            # Alternative to the loop below:
            # sns.boxplot(df_metric, x="roi", y="score", ax=ax)

            # Prepare ROI-wise medians for correlation with NCSNR
            metric_mean_per_roi: dict[str, float] = {}
            for r_idx, roi in enumerate(rois):
                df_roi = df_metric[df_metric["roi"] == roi]
                if len(df_roi) != len(subjects):
                    # try to remove duplicates:
                    df_roi = df_roi.drop_duplicates(
                        subset=["subject", "score", "layer", "roi"]
                    )
                # lower, upper = np.percentile(df_roi["score"], [ql, qu],
                #     method="closest_observation")
                lower = df_roi["score"].min()
                upper = df_roi["score"].max()
                mean = df_roi["score"].mean()
                metric_mean_per_roi[roi] = float(mean)

                x0, x1 = r_idx - 0.4, r_idx + 0.4
                # IQR band
                ax.fill_between(
                    [x0, x1],
                    [lower, lower],
                    [upper, upper],
                    color=colors[metric],
                    alpha=0.25,
                    linewidth=0,
                )

                # 25th/75th lines
                ax.hlines(
                    [lower, upper],
                    x0,
                    x1,
                    colors=colors[metric],
                    linestyles="--",
                    linewidth=1,
                    alpha=0.5,
                )

                # Mean segment
                ax.hlines(mean, x0, x1, colors=colors[metric], linewidth=2)


            # add a legend with the color of each metric
            if metric_idx == len(metrics) - 1:
                handles = [
                    Line2D(
                        [0], [0], color=colors[metric], linewidth=3,
                        label=metric.replace("_", " ").title()
                    )
                    for metric in metrics
                ]
                ax.legend(handles=handles, title="Metric", frameon=False,
                             loc="upper right")

            # If available, annotate Pearson correlation between
            #  metric medians and NCSNR
            if compare_to_ncsnr:
                try:
                    x_vals = np.array(
                        [metric_mean_per_roi[roi] for roi in rois], dtype=float
                    )
                    y_vals = np.array(
                        [ncsnr_median_per_roi[roi] for roi in rois], dtype=float
                    )
                    if (
                        np.all(np.isfinite(x_vals))
                        and np.all(np.isfinite(y_vals))
                        and np.std(y_vals) > 0
                    ):
                        # r = float(np.corrcoef(x_vals, y_vals)[0, 1])
                        # # off-diag element
                        r = float(pearsonr(x_vals, y_vals).statistic)
                        ax.text(
                            0.98,
                            0.05,
                            f"Corr. with NCSNR: r={r:.2f}",
                            transform=ax.transAxes,
                            ha="right",
                            va="bottom",
                            fontsize=9,
                            bbox=dict(
                                boxstyle="round",
                                facecolor="white",
                                alpha=0.6,
                                linewidth=0,
                            ),
                        )
                except Exception as e:
                    LOGGER.warning(f"Failed to compute correlation with NCSNR: {e}")

        metric_names = [metric.replace("_", " ").capitalize().replace("Rsa", "RSA")\
                        for metric in metrics]
        ax.set_title(f"{metric_name} and " + f" | split={split}")
        # (f"Mean of similarity between leave-one-out subject
            # mean and remaining subject")
        ax.set_xlabel("ROI")
        ax.set_ylabel(f"{' / '.join(metric_names)}")
        ax.set_xticks(range(len(rois)))
        ax.set_xticklabels(rois)
        ax.set_xlim(-0.5, len(rois) - 0.5)
        ax.set_ylim(0, 1)
        ax.set_title(f"{' and '.join(metric_names)}" + f" | split={split}")

        # If comparing to NCSNR, draw a top subplot with
        # ROI-wise NCSNR medians and IQRs
        if compare_to_ncsnr:
            ax_ncsnr = axes[-1]
            # For consistency with noise ceiling panels,
            # draw per-ROI median and IQR across subjects
            for r_idx, roi in enumerate(rois):
                # Recompute from per-subject medians to
                # capture spread across subjects
                subj_vals = [
                    float(
                        np.nanmean(
                            get_roi_wise_voxel_consistency(
                                fmri_cfg, roi_masks, str(subject_id)
                            )[roi]
                        )
                    )
                    for subject_id in subjects
                ]
                # lower, upper = np.percentile(subj_vals, [ql, qu])
                lower = np.nanmin(subj_vals)
                upper = np.nanmax(subj_vals)
                mean = np.nanmean(subj_vals)
                x0, x1 = r_idx - 0.4, r_idx + 0.4
                ax_ncsnr.fill_between(
                    [x0, x1],
                    [lower, lower],
                    [upper, upper],
                    color=colors_rois[r_idx],
                    alpha=0.25,
                    linewidth=0,
                )
                ax_ncsnr.hlines(
                    [lower, upper],
                    x0,
                    x1,
                    colors=colors_rois[r_idx],
                    linestyles="--",
                    linewidth=1,
                    alpha=0.5,
                )
                ax_ncsnr.hlines(mean, x0, x1, colors=colors_rois[r_idx], linewidth=2)

            ax_ncsnr.set_title("NCSNR (mean across subjects)")
            ax_ncsnr.set_xlabel("ROI")
            ax_ncsnr.set_ylabel("NCSNR")
            ax_ncsnr.set_xticks(range(len(rois)))
            ax_ncsnr.set_xticklabels(rois)
            ax_ncsnr.set_xlim(-0.5, len(rois) - 0.5)

        # plt.suptitle("Mean of similarity between leave-one-out
        #  subject mean and remaining subject.\n"+\
        #             f"Shaded area indicates range across subjects.")
        plt.tight_layout()
        plt.savefig(plot_path.parent / "plots" / outfilename)
        plt.close()
        LOGGER.info(f"Saved plot to {plot_path.parent / 'plots' / outfilename}")

    # * Also create a plot showing the MSEs and scores,
    # * to see overfitting magnitude
    # (If that data is available)
    if "ridgecv_val_mse" in df.columns:
        # Also plot debug plots showing train and validation and
        #  ridge-cv validation MSEs, just for linear predictivity.
        train_df = df[df["split"] == "train"]

        df_metric = train_df[train_df["metric"] == "linear_predictivity"]
        # reduce to noise ceilings only:
        df_metric = df_metric[df_metric["layer"] == df_metric["roi"]]
        df_metric = df_metric.drop_duplicates(
                subset=["subject", "score", "layer", "roi"]
        )

        # Plot boxplots across subjects, per ROI:

        mse_map = {
            "ridgecv_val_mse": "Ridge-CV MSE (validation points)",
            "train_set_train_split_mse": "MSE on train splits",
            "train_set_val_split_mse": "MSE of validation splits",
            "train_set_train_split_score_mean": "Score on train splits",
            "score": "Score",
        }
        palette_map = {
            "Ridge-CV MSE (validation points)": "darkred",
            "MSE on train splits": "lightblue",
            "MSE of validation splits": "darkblue",
            "Score on train splits": "lightblue",
            "Score": "darkblue",
        }

        # convert to long format to use an auto-legend (hue by "mse_type_key")
        df_long = df_metric.melt(
            id_vars=["roi", "subject"],
            value_vars=list(mse_map.keys()),
            var_name="mse_type_key",
            value_name="mse_or_score",
        )
        df_long["mse_type_title"] = df_long["mse_type_key"].map(mse_map)
        df_long_mse = df_long[df_long["mse_type_key"].isin(
            [ "ridgecv_val_mse",
            "train_set_train_split_mse", "train_set_val_split_mse"]
        )]
        df_long_score = df_long[df_long["mse_type_key"].isin(
            ["score", "train_set_train_split_score_mean"]
        )]

        nrows = 2
        fig, axes = plt.subplots(figsize=(15, 10), nrows=nrows, ncols=1)
        ax=axes[0]

        sns.boxplot(
            data=df_long_mse, x="roi", y="mse_or_score", hue="mse_type_title",
            palette=palette_map, ax=ax
        )
        ax.grid(True)
        ax.set_title("Different types of MSEs for linear predictivity, across ROIs")
        ax.set_xlabel("ROI")
        ax.set_ylabel("MSE")
        ax.set_xticks(range(len(rois)))
        ax.set_xticklabels(rois)
        ax.set_xlim(-0.5, len(rois) - 0.5)
        ax.legend()

        ax = axes[1]
        sns.boxplot(
            data=df_long_score, x="roi", y="mse_or_score", hue="mse_type_title",
            palette=palette_map, ax=ax
        )
        ax.grid(True)
        ax.legend()
        ax.set_title("Score on train and val splits of train set, across ROIs")
        ax.set_xlabel("ROI")
        ax.set_ylabel("R^2 score")
        ax.set_xticks(range(len(rois)))
        ax.set_xticklabels(rois)
        ax.set_xlim(-0.5, len(rois) - 0.5)


        plt.tight_layout()
        plt.savefig(plot_path.parent / "plots" / "mse_vs_alpha.png")
        plt.close()
        LOGGER.info(f"Saved plot to {plot_path.parent / 'plots' / 'mse_vs_alpha.png'}")

if __name__ == "__main__":
    plot_noiseceilings()
