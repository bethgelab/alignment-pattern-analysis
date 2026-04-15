#!/usr/bin/env python3
"""Generate figure 0 for boldmoments analysis.

For a given metric x main_outcome, figure 0 consists of 1 panel showing the
avg. model performance (SD across ROIs).
"""

import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from multitasking.analyses.dataframe_utils import get_order
from multitasking.CONSTANTS import ROIS as rois
from multitasking.plot_creation.plot_utils import save_this, set_ticks_0_1

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--metric",
    type=str,
    default="rsa",
    help="Metric to use (default: rsa)",
)
@click.option(
    "--main-outcome",
    type=click.Choice(["score", "score_norm_upper", "score_norm_lower"]),
    default="score",
    help="Main outcome variable (default: score)",
)
@click.option(
    "--split",
    type=str,
    default="test",
    help="Data split (default: test)",
)
@click.option(
    "--plot-path",
    type=click.Path(exists=False, file_okay=False, path_type=Path),
    default=None,
    help="Output directory for plots (default: results/boldmoments/plots/YYYYMMDD)",
)
@click.option(
    "--taskonomy",
    type=bool,
    default=False,
    help="Whether to include taskonomy models (default: False)",
)
def main(metric, main_outcome, split, plot_path, taskonomy):
    scratch = Path(os.environ.get("SCRATCH", "/scratch"))
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    date_str = datetime.now().strftime("%Y%m%d")
    if plot_path is None:
        plot_path = project_root / "results" / "boldmoments" / "plots" / date_str
    plot_path.mkdir(parents=True, exist_ok=True)

    # Load data
    subject_avg_df_model_order = pickle.load(
        open(scratch / "subject_avg_df_equiv_score_norm_lower.pkl", "rb")
    )
    subject_avg_df_model_order["model_provider"] = subject_avg_df_model_order["model"].str.split("/").str[0]
    subject_avg_df_model_order = subject_avg_df_model_order[
        (subject_avg_df_model_order["split"] == split)
        & (subject_avg_df_model_order["metric"] == "rsa")
        & (subject_avg_df_model_order["is_best_layer"])
        & ((subject_avg_df_model_order["model_provider"] != "taskonomy") if not taskonomy else True)
    ]

    model_order = get_order(
        subject_avg_df_model_order,
        column_name="score_norm_lower_mean",
        metric="rsa",
        order_by="model")

    subject_avg_df = pickle.load(
        open(scratch / f"subject_avg_df_equiv_{main_outcome}.pkl", "rb")
    )

    subject_avg_df["model_provider"] = subject_avg_df["model"].str.split("/").str[0]
    subject_avg_df["predictor"] = (
        subject_avg_df["model"].astype(str)
        + "__"
        + subject_avg_df["layer"].astype(str)
    )

    # Model colors
    with open(project_root / "indiv_model_colors_dict.json") as f:
        model_colors_dict = json.load(f)

    # Model legend mapping
    with open(project_root / "model_legend_mapping.json") as f:
        model_legend_mapping = json.load(f)

    # Filter data
    panel_1_df_wo_taskonomy = subject_avg_df[
        (subject_avg_df["split"] == split)
        & (subject_avg_df["metric"] == metric)
        & (subject_avg_df["is_best_layer"])
        & (subject_avg_df["model_provider"] != "taskonomy")
    ]
    panel_1_df_taskonomy = subject_avg_df[
        (subject_avg_df["split"] == split)
        & (subject_avg_df["metric"] == metric)
        & (subject_avg_df["is_best_layer"])
    ]

    if taskonomy:
        panel_1_df = panel_1_df_taskonomy
    else:
        panel_1_df = panel_1_df_wo_taskonomy


    panel_1_df = (
        panel_1_df.set_index("model")
        .loc[model_order]
        .reset_index()
    )
    table = panel_1_df_taskonomy.copy()
    table["model"] = table["model"].apply(lambda x: model_legend_mapping[x].replace("_", "-"))
    model_order_renamed = [
        model_legend_mapping[m].replace("_", "-")
        for m in model_order
    ]
    table = table.set_index("model").loc[model_order_renamed].reset_index()
    table["model"] = pd.Categorical(table["model"], categories=model_order_renamed, ordered=True)
    table = table.pivot(index="model", columns="roi", values=f"{main_outcome}_mean")
    table = table.reindex(columns=rois)

    styled = (
        table.style
        .format("{:.2f}")
        .highlight_max(axis=0, props="textbf:--rwrap;")
    )

    styled.to_latex(plot_path / f"figure_0_{metric}_{main_outcome}.tex",

                                    hrules=True,
                                    column_format="l" + "r" * len(styled.columns),)

    # Create figure
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 8,
        "lines.linewidth": 1,
        "lines.markersize": 5,
        "font.family": "serif",
    })
    fig_width = 5.5 if taskonomy else 2.75
    plt.figure(figsize=(fig_width, 1))
    ax = plt.gca()

    # sns.barplot(
    #     data=panel_1_df,
    #     x="model",
    #     y=f"{main_outcome}_mean",
    #     errorbar="sd",
    #     palette=model_colors_dict,
    #     ax=ax,
    #     saturation=1,
    # )

    import numpy as np

    # Extract values in deterministic order
    colors = [model_colors_dict[m] for m in model_order]
    x = np.arange(len(model_order))
    means = []
    sds = []
    for m in model_order:
        values = panel_1_df.loc[panel_1_df["model"] == m, f"{main_outcome}_mean"].values
        means.append(values.mean())
        sds.append(values.std())
    for i, mean, sd in zip(x, means, sds, strict=False):
        ax.bar(
            i,
            mean,
            capsize=0,
            color=colors[i],
        )
        ax.errorbar(
            i,
            mean,
            yerr=np.array([sd, sd])[:, None],
            fmt="none",
            ecolor="k",
            alpha=1,
            capsize=0,
        )

    if "norm" in main_outcome:
        ax.axhline(y=1, color="black", linestyle="--", linewidth=1)

    ax.set_xticks(
        range(len(model_order)),
        [model_legend_mapping[model].replace("_", "-") for model in model_order],
        rotation=90,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    if main_outcome == "score_norm_lower":
        ax.set_ylim(0, 1.2)

    if "norm" in main_outcome:
        set_ticks_0_1(ax, axis="y", step= 0.4)
    else:
        set_ticks_0_1(ax, axis="y", step= 0.2)
    sns.despine(ax=ax)

    save_this(
        path=plot_path,
        fname=f"figure_0_{metric}_{main_outcome}_{taskonomy}",
        dpi=300,
        transparent=False,
    )
    logger.info(f"Saved figure to {plot_path}/figure_0_{metric}_{main_outcome}_{taskonomy}.*")


if __name__ == "__main__":
    main()
