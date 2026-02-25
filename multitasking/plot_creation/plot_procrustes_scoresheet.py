"""Plot the procrustes scoresheet.

This might fit better into into utils/plotting.py.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_procrustes_scoresheet(scoresheet, config, n_samples=None, plot_dir="."):
    """Expects list of dicts of the format below.

    scoresheet = {
        "score": (number),
        "model_str": model_str(config),
        "model_layer": layer,
        "model_layer_index": layer_index,
        "region": roi_name,
        "score_set": "train" or "test",
        "train_type": "procrustes",
        "procrustes_score_function": score_function,
        "projection_strategy": feature_reduction_method,
        "timestamp": time.time(),
        # "n_samples": n_samples_alignment_str,
        "distance_1": "N/A",
        "distance_2": "N/A",
        ...
    }
    """
    df_full = pd.DataFrame(scoresheet)
    # if n_samples is not None:
    #     df_full = df_full[df_full["n_samples"] == n_samples]

    procrustes_score_funs = df_full["procrustes_score_function"].unique()
    projection_strategies = df_full["projection_strategy"].unique()
    # n_samples_list = df_full["n_samples"].unique()

    for procrustes_score_fun in procrustes_score_funs:
        for projection_strategy in projection_strategies:
            #   for n_samples in n_samples_list:

            # Create one plot with two subplots for train and test
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            for i, split in enumerate(["train", "test"]):
                df = df_full[df_full["score_set"] == split]
                df = df[df["procrustes_score_function"] == procrustes_score_fun]
                df = df[df["projection_strategy"] == projection_strategy]
                # df = df[df["n_samples"] == n_samples]
                df = df.drop(
                    columns=[
                        "score_set",
                        "train_type",
                        "distance_1",
                        "distance_2",
                        "model_layer_index",
                        "model_layer",
                        "timestamp",
                        # "n_samples",
                        "procrustes_score_function",
                        "projection_strategy",
                    ]
                )

                # For each region, plot a scatterplot of the scores. Use seaborn.
                sns.scatterplot(
                    data=df, x="model_str", y="score", hue="region", ax=axs[i]
                )
            # axes labels, legend, title
            axs[0].set_xlabel("Region")
            axs[0].set_ylabel("Score")
            axs[0].set_title("Train")
            axs[1].set_title("Test")
            plt.legend()
            plt.savefig(
                f"{plot_dir}/procrustes_scores_{procrustes_score_fun}_{projection_strategy}.png"
            )
