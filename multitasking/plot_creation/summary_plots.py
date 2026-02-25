import logging
import os
import pickle
from pathlib import Path

import click
import matplotlib.pyplot as plt
import seaborn as sns

from .plot_utils import (
    add_noise_ceiling,
    ci_plots,
    create_base_box_plot,
    rotate_and_truncate_xticks,
    save_this,
)

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "--split",
    "split",
    type=str,
    default="test",
    help="one of: 'train', 'test', or both'",
)
@click.option(
    "--plot-path",
    "plot_path",
    type=click.Path(exists=False, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--format",
    "formats",
    type=click.Choice(["pdf", "png", "svg"], case_sensitive=False),
    multiple=True,
    default=['png'],
    help="Output formats (choose 0 to 3)."
)
def plot_results(
    split: str,
    plot_path: Path,
    formats: tuple[str, ...],
):
    """Plot the summary of the results.

    Given a list of output directories, pulls in the results from scoresheets stored
    in these directories and creates a plot alignment per ROI vs. models.
    """
    SCRATCH = Path(os.environ.get("SCRATCH", "/scratch"))
    # Load intersubject dataframes (up to 2)
    with open(SCRATCH / "bootstrap_ci.pkl", "rb") as f:
        bootstrap_ci = pickle.load(f)
    with open(SCRATCH / "intersubject_df.pkl", "rb") as f:
        intersubject_df = pickle.load(f)
    with open(SCRATCH / "pairwise_subject_df.pkl", "rb") as f:
        pairwise_subject_df = pickle.load(f)
    with open(SCRATCH / "df.pkl", "rb") as f:
        df = pickle.load(f)



    ############## Plotting ####################
    plot_path.mkdir(parents=True, exist_ok=True)
    metric_palette = ['darkblue', 'darkred', 'darkgreen']
    metric_color_dict = dict(zip(
        ["linear_predictivity", "rsa", "model_brain_aps"], metric_palette,
                                strict=False))

    ####### Create split by metric plots #######
    for metric in ["rsa", "linear_predictivity"]:
        # overall

        split_df = df[(df["split"] == split)]
        ax = create_base_box_plot(
            df=split_df,
            x=f"model_order_{metric}",
            y="score_mean",
            hue="metric",
            palette=metric_palette,
            boxprops=dict(edgecolor="none"),
        )
        rotate_and_truncate_xticks(ax)
        sns.despine()

        total_length = len(ax.get_xticks())
        for i, xpos in enumerate(ax.get_xticks()):
            ax.annotate(text = total_length - i,
                xy = (xpos-0.5, 0.4),
                xytext = (xpos-0.5, 0.4),
                zorder=10,
                fontsize=8)
            if i % 2 == 0:  # every other band
                ax.axvspan(i-0.5, i+0.5, facecolor="darkgray", alpha=0.4, zorder=0,
                            edgecolor="none")

        save_this(
            plot_path,
            fname=f"{metric}_{split}",
            formats=formats,
            fig=plt.gcf(),
            transparent=False,
        )
        plt.close()


    ###### Create CI overlap plots #######


    n_overlap_metric_roi = {metric: {roi: 0 for roi in df["roi"].unique()} \
        for metric in ["rsa", "linear_predictivity"]}

    split="test"
    separate_plots = False
    if separate_plots:
        raise NotImplementedError("Separate plots are not implemented yet")
        # for metric in ["rsa", "linear_predictivity"]:
        #     for roi in df["roi"].unique():
        #         plt.figure(figsize=(4, 4))
        #         ax = ci_plots(df, roi=roi, boot_cis=bootstrap_ci, metric=metric,
        #                       model_order=model_orders[f'{metric}_{roi}'],
        #                       n_overlap_metric_roi=n_overlap_metric_roi,
        #                       ax=plt.gca(), xoffset=.2, metric_color_dict=metric_color_dict

        #                       )
        #         models = df[~df['model'].str.contains("taskonomy")]['model'].unique()
        #         ax.set_xticks(range(len(models)), model_orders[f'{metric}_{roi}'].index(models),
        #                 fontsize=8,
        #                 rotation=90)

        #         # Add noise ceilings for each intersubject dataframe
        #         for intersubject_df, ls in zip(intersubject_dfs, ["--", "-."]):

        #             add_noise_ceiling(ax, intersubject_df[intersubject_df['layer']==roi],
        #                             roi, split, metric, var_type="sem",
        #                             color=metric_color_dict[metric], linestyle=ls)

        #         sns.despine()

        #         plt.title(f"{roi}\n"
        #                 f"{metric}: {n_overlap_metric_roi[metric][roi]}")

        #         rotate_and_truncate_xticks(ax, truncation=30)
        #         save_this(plot_path, fname=f"{roi}_{metric}_ci_overlap", formats=['png', 'svg'],
        #                 transparent=False
        #                 )
        #         plt.close()
    else:
        for roi in df["roi"].unique():
            # without taskonomy models
            plt.figure(figsize=(4, 3))
            metric="rsa"
            models = df[~df['model'].str.contains("taskonomy")]['model'].unique()
            model_order = [m for m in list(df[f"model_order_{metric}"].cat.categories) if m
                           in models]
            ax = ci_plots(df, roi=roi, boot_cis=bootstrap_ci, metric="rsa",
                    model_order=model_order,
                    n_overlap_metric_roi=n_overlap_metric_roi,
                    ax=plt.gca(), xoffset=.2, metric_color_dict=metric_color_dict
                    )

            ax= ci_plots(df, roi=roi, boot_cis=bootstrap_ci, metric="linear_predictivity",
                                    n_overlap_metric_roi=n_overlap_metric_roi,
                                    xoffset=-.2,
                                    metric_color_dict=metric_color_dict,
                    model_order=model_order,
                    ax=ax
                    )

            ax.set_xticks(range(len(model_order)), model_order,
                        fontsize=8,
                        rotation=90)

            total_length = len(ax.get_xticks())
            for i in range(total_length):

                if i % 2 == 1:  # every other band
                    ax.axvspan(i-0.5, i+0.5, facecolor="darkgray", alpha=0.4, zorder=0,
                            edgecolor="none")

            # Add noise ceilings for each intersubject dataframe
            for noise_ceiling_df, ls in zip(
                [intersubject_df, pairwise_subject_df], ["--", "-."], strict=False):
                for metric, color in metric_color_dict.items():
                    add_noise_ceiling(ax, noise_ceiling_df[noise_ceiling_df['layer']==roi],
                                    roi, split, metric, var_type="sem", color=color, linestyle=ls)

            sns.despine()
            ax.set_ylabel("Alignment score")
            ax.set_xlabel("Model")

            plt.title(f"{roi}\n"
                    f"RSA: {n_overlap_metric_roi['rsa'][roi]}, "
                    f"LP: {n_overlap_metric_roi['linear_predictivity'][roi]}")

            rotate_and_truncate_xticks(ax, truncation=30)
            save_this(plot_path, fname=f"{roi}_ci_overlap", formats=['png', 'svg', 'pdf'],
                    transparent=False
                    )
            plt.close()


            # with taskonomy models
            plt.figure(figsize=(8, 3))
            metric="rsa"
            models = df['model'].unique()

            model_order = [m for m in list(df[f"model_order_{metric}"].cat.categories) if m
                           in models]
            ax = ci_plots(df, roi=roi, boot_cis=bootstrap_ci, metric="rsa",
                    model_order=model_order,
                    n_overlap_metric_roi=n_overlap_metric_roi,
                    ax=plt.gca(), xoffset=.2, metric_color_dict=metric_color_dict
                    )

            ax= ci_plots(df, roi=roi, boot_cis=bootstrap_ci, metric="linear_predictivity",
                                    n_overlap_metric_roi=n_overlap_metric_roi,
                                    xoffset=-.2,
                                    metric_color_dict=metric_color_dict,
                    model_order=model_order,
                    ax=ax
                    )

            ax.set_xticks(range(len(model_order)), model_order,
                        fontsize=8,
                        rotation=90)
            total_length = len(ax.get_xticks())
            for i in range(total_length):
                if i % 2 == 0:  # every other band
                    ax.axvspan(i-0.5, i+0.5, facecolor="darkgray", alpha=0.4, zorder=0,
                            edgecolor="none")

            # Add noise ceilings for each intersubject dataframe
            for noise_ceiling_df, ls in zip(
                [intersubject_df, pairwise_subject_df], ["--", "-."], strict=False):
                for metric, color in metric_color_dict.items():
                    add_noise_ceiling(ax, noise_ceiling_df[noise_ceiling_df['layer']==roi],
                                    roi, split, metric, var_type="sem", color=color, linestyle=ls)

            sns.despine()
            ax.set_ylabel("Alignment score")
            ax.set_xlabel("Model")

            plt.title(f"{roi}\n")

            rotate_and_truncate_xticks(ax, truncation=30)
            save_this(plot_path, fname=f"{roi}_ci_overlap_taskonomy", formats=['png', 'svg'],
                    transparent=False
                    )
            plt.close()
if __name__ == "__main__":
    plot_results()
