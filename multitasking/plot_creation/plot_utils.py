import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
import matplotlib.patches as patches
import numpy as np
import seaborn as sns

import logging
logger = logging.getLogger(__name__)

def save_this(path, fname, formats=None, fig=None, transparent=True, dpi=300):
    if formats is None:
        formats = ["png", "svg", "pdf"]  # default formats

    if fig is None:
        fig = plt.gcf()

    for fmt in formats:
        full_path = os.path.join(path, f"{fname}.{fmt}")
        # make the directory if it doesn't exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        fig.savefig(
            full_path, dpi=dpi, transparent=transparent, bbox_inches="tight", format=fmt
        )


def create_base_box_plot(
    df, x, y, hue=None, ax=None, palette="deep", figsize=(10, 4), **kwargs
):
    """Create a seaborn boxplot, return the axis."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax, palette=palette, 
                medianprops=dict(color="white"), 
                flierprops=dict(marker="d", color="black", 
                                alpha=0.5, markersize=3),
                **kwargs)
    # match whiskers & caps to box fill colors
    for _patch, color in zip(ax.artists, palette, strict=False):
        # Whiskers and caps come after each box (2 whiskers + 2 caps)
        for line in ax.lines[:4]:
            line.set_color(color)
        ax.lines = ax.lines[4:]  # trim processed lines

    

    return ax


def create_column_wise_scatter(
    full_df,
    column_key,
    column_value,
    x,
    y,
    add_errorbars=True,
    ax=None,
    figsize=(10, 4),
    color="k",
    **kwargs,
):
    """Create a scatter plot for a specific value in a given column.

    Parameters
    ----------
    full_df : pd.DataFrame
        The full DataFrame containing the data.
    column_key : str
        The column to filter on (e.g., "roi" or "model").
    column_value : str
        The value in `column_key` to filter for.
    x : str
        The column to use for the x-axis.
    y : str
        The column to use for the y-axis.
    add_errorbars : bool, optional
        Whether to add error bars to the scatter points (default: True).
    ax : matplotlib.axes.Axes, optional
        The axis to plot on. If None, a new figure and axis are created.
    figsize : tuple, optional
        The size of the figure if a new one is created (default: (10, 4)).
    color : str, optional
        The color of the scatter points (default: 'k').

    Returns:
    -------
    ax
        the axis.
    """
    df = full_df[full_df[column_key] == column_value]
    if column_key == "roi":
        row_key = "model"
    elif column_key == "model":
        row_key = "roi"
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    # Create the scatterplot first so seaborn assigns colors
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        ax=ax,
        color=color,
        # style="is_best_score",
        zorder=5,
        **kwargs,
    )

    if add_errorbars:
        metric_color_dict = kwargs.get("metric_color_dict", None)
        if metric_color_dict is None:
            metric_color_dict = dict(zip(
                ["linear_predictivity", "rsa", "model_brain_aps"], 
                ['darkblue', 'darkred', 'darkgreen']
            ))

        for _, row in df.iterrows():
            ax.errorbar(
                x=row[row_key],
                y=row["score_mean"],
                yerr=row["score_std"],
                fmt="none",
                capsize=3,
                zorder=4,
                color=metric_color_dict[row["metric"]],
            )
    ax.set_title(column_value)
    return ax


def set_ticks_0_1(ax=None, start=0, step=0.2, stop =1, axis="y"):
    """Set y-axis ticks at step intervals (e.g. .2, .4, .6, .8). Does not change the y-axis range."""
    if ax is None:
        ax = plt.gca()
    if axis == "y":
        ylim = ax.get_ylim()
        ticks = np.arange(start, stop, step)
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FixedFormatter([f"{t:.1f}" for t in ticks]))
        ax.set_ylim(ylim)
    elif axis == "x":
        xlim = ax.get_xlim()
        ticks = np.arange(start, stop, step)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([f"{t:.1f}" for t in ticks]))
        ax.set_xlim(xlim)


def rotate_and_truncate_xticks(ax, truncation=30):
    xlabels = [tick.get_text() for tick in ax.get_xticklabels()]
    ax.set_xticklabels([xlabel[:truncation] for xlabel in xlabels], rotation=90)


def add_shading(df, roi, ax, color_mapping):
    # Get mapping from category to x-position
    xticks = ax.get_xticks()
    xlabels = [tick.get_text() for tick in ax.get_xticklabels()]
    xname_to_pos = dict(zip(xlabels, xticks, strict=False))

    # Add shading for max rows
    df = df[df["roi"] == roi]
    for metric in ["rsa", "linear_predictivity"]:
        sub_df = df[df["metric"] == metric]
        for name in sub_df.loc[sub_df["is_best_score"], "model"]:
            x_pos = xname_to_pos.get(name)
            if x_pos is not None:
                plt.axvspan(
                    x_pos - 0.4,
                    x_pos + 0.4,
                    color=color_mapping[metric],
                    alpha=0.1,
                    zorder=-1,
                    edgecolor='none',
                )


def add_noise_ceiling(ax, intersubject_df, roi, split, metric, var_type="std", **kwargs):
    """Add a hline showing the noise ceiling to the plot, to the given axis."""
    intersubject_df_roi = intersubject_df[intersubject_df["roi"] == roi]
    intersubject_df_roi = intersubject_df_roi[intersubject_df_roi["split"] == split]
    intersubject_df_roi = intersubject_df_roi[intersubject_df_roi["metric"] == metric]
    n_rows_intersubject = len(intersubject_df_roi)
    n_subjects = intersubject_df_roi["subject"].nunique()
    # assert n_rows_intersubject == n_subjects, (
    #     f"Expected {n_subjects} rows in intersubject df,"
    #     f"found {n_rows_intersubject}"
    #     " (did you add another factor of variation?)"
    # )
    noise_ceiling_mean = intersubject_df_roi["score"].mean()
    n_vals = intersubject_df_roi["subject"].nunique()
    if var_type == "std":
        noise_ceiling_std = intersubject_df_roi["score"].std(ddof=1)
    elif var_type == "sem":
        # TODO: check if this is correct
        logger.info(f"Calculating noise ceiling SEM with {n_vals} values")
        noise_ceiling_std = intersubject_df_roi["score"].std(ddof=1) / np.sqrt(n_vals)
        
    # Get current x-limits from the axes
    _xmin, _xmax = ax.get_xlim()
    xmin = kwargs.get("xmin", _xmin)
    xmax = kwargs.get("xmax", _xmax)

    # Horizontal line
    ax.hlines(
        y=noise_ceiling_mean,
        xmin=xmin,
        xmax=xmax,
        colors=kwargs.get("color", "black"),
        linestyles=kwargs.get("linestyle", "--"),
        linewidth=1,
        label="Noise ceiling",
    )

    # Optional shaded band
    if n_rows_intersubject > 1:
        ax.fill_between(
            [xmin, xmax],
            noise_ceiling_mean - noise_ceiling_std,
            noise_ceiling_mean + noise_ceiling_std,
            color=kwargs.get("color", "grey"),
            alpha=0.1,
            zorder=-2,
            linewidth=0,
        )


def ci_plots(df, roi, boot_cis, n_overlap_metric_roi, metric, model_order, ax, 
             xoffset=0, metric_color_dict=None):
    best_model_roi = df['model'].iloc[df[f"model_order_{metric}_{roi}"].cat.codes.idxmax()]
    lower_of_best, _ = boot_cis[metric][roi][best_model_roi]
    plt.sca(ax)
    for model in model_order:
        m = df[(df['roi']==roi) &
               (df['model']==model) &
               (df['split']=='test') & 
               (df['metric']==metric)
               ]['score_mean'].to_numpy()
        ci_lower, ci_upper = boot_cis[metric][roi][model]
        if m >= lower_of_best:
            marker = "*" 
            color=metric_color_dict[metric]
            n_overlap_metric_roi[metric][roi] += 1
            alpha=1
            s=40
        else:
            marker="o"
            color=metric_color_dict[metric]
            alpha=.3
            s=20
        plt.scatter(
            model_order.index(model)+xoffset, 
                     y= m,
                     color=color,
                     edgecolor='none',
                     alpha=alpha,
                     s=s,
                     marker=marker)
        plt.errorbar(
            model_order.index(model)+xoffset, 
                     y= (ci_upper + ci_lower)/2,
                     yerr=(ci_upper - ci_lower)/2,
                     color=metric_color_dict[metric],
                    alpha=alpha,
                     zorder=0
                     )
        # if model == best_model_roi:
        #     plt.hlines(y=ci_lower, xmin=0, xmax=len(models), color=metric_color_dict[metric], zorder=0)
    return ax