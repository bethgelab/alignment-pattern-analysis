import logging
import os
import pickle
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

LOGGER = logging.getLogger(__name__)


def save_this(path, fname, formats=None, fig=None, transparent=True):
    if formats is None:
        formats = ["png", "svg", "pdf"]  # default formats

    if fig is None:
        fig = plt.gcf()

    for fmt in formats:
        full_path = os.path.join(path, f"{fname}.{fmt}")
        fig.savefig(
            full_path, dpi=300, transparent=transparent, bbox_inches="tight", format=fmt
        )


@click.command()
@click.option(
    "--output-supdir",
    "output_supdir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--output-dirs",
    "output_dirs",
    multiple=True,
    default=[],
)
@click.option("--filename", "filename", type=str, required=True)
@click.option(
    "--results-dir",
    "results_dir",
    type=click.Path(exists=False, file_okay=False, path_type=Path),
    required=True,
)
@click.option("--results-subdir", "results_subdir", type=str, default="")
@click.option("--plot-fname", "plot_fname", type=str, default="latest_results")
@click.option(
    "--split",
    "split",
    type=str,
    required=True,
    help="one of: 'train', 'test', or 'both'",
)
@click.option("--procrustes/--no-procrustes", "procrustes", type=bool, default=False)
@click.option("--format", "res_format", type=str, default="new")
def plot_results(
    output_supdir: Path,
    output_dirs: list[str],
    filename: str,
    results_dir: Path,
    results_subdir: str,
    split: str,
    plot_fname: str = "latest_results",
    procrustes: bool = False,
    res_format: str = "new",
):
    """Plot the alignment of models vs. ROIs.

    Given a list of output directories, pulls in the results from scoresheets stored
    in these directories and creates a plot alignment per ROI vs. models.
    """
    if len(output_dirs) == 0:
        output_dirs = os.listdir(output_supdir)
    all_scoresheets = []
    n = 0
    # filename = "scoresheet.pkl" if not procrustes else "scoresheet_procrustes.pkl"

    for output_dir in output_dirs:
        # check if its a directory
        if not os.path.isdir(os.path.join(output_supdir, output_dir)):
            continue
        if res_format == "old":  # supporting pkl files
            fpath = Path(
                os.path.join(output_supdir, output_dir, "scoresheets", filename)
            )
            if fpath.exists():
                with open(fpath, "rb") as f:
                    scoresheet = pickle.load(f)
                    all_scoresheets.extend(scoresheet)
                    n += 1
            else:
                f"Couldn't load scoresheet for run {output_dir}"

        elif res_format == "new":
            LOGGER.info(f"Loading scoresheet for run {output_dir}")
            os.listdir(os.path.join(output_supdir, output_dir))
            fpath = Path(
                os.path.join(
                    output_supdir,
                    output_dir,
                    "scoresheets",
                    filename,
                )
            )
            if fpath.exists():
                scoresheet = pd.read_csv(fpath)
                all_scoresheets.append(scoresheet)
                n += 1
            else:
                LOGGER.info(f"Couldn't load scoresheet for run {output_dir}")

    if len(all_scoresheets) >= 1 and res_format == "old":
        # Step 1: Load the list of dicts into a DataFrame
        df = pd.DataFrame(all_scoresheets)
        LOGGER.info("STep 1: Loaded scoresheets into DataFrame")
        LOGGER.info(f"Loaded {len(df)} rows from {n} runs.")

        # Step 2: Rename fields for clarity (optional)
        df = df.rename(
            columns={
                "region": "roi",
                "train_type": "metric",
                "score_set": "split",
                "model_str": "model",
                "layer_str": "layer",
            }
        )

        LOGGER.info("Step 2: Renamed columns for clarity")

    elif len(all_scoresheets) >= 1 and res_format == "new":
        df = pd.concat(all_scoresheets)
    else:
        return

    # Step 3: Add ROI stream info based on roi_name (you need to define this mapping)
    early = {"V1", "V2", "V3"}
    dorsal_rois = {
        "V3A",
        "V3B",
        "V6",
        "V6A",
        "V7",
        "MT",
        "MST",
        "FST",
        "LO1",
        "LO2",
        "LO3",
        "RSC",
        "IPS1",
    }
    ventral_rois = {"V4", "V8", "FFC", "PIT"}

    def classify_stream(region):
        if region in dorsal_rois:
            return "dorsal"
        elif region in ventral_rois:
            return "ventral"
        elif region in early:
            return "early"
        else:
            return "unknown"  # or np.nan

    df["stream"] = df["roi"].apply(classify_stream)
    # create results directory if it doesn't exist
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(os.path.join(results_dir, "results_df.pkl"))

    roi_hierarchy_map = {
        "V1": 1,
        "V2": 2,
        "V3": 3,
        "V3A": 3,
        "V3B": 3,
        "V4": 4,
        "V6": 6,
        "V6A": 6,
        "V7": 7,
        "V8": 8,
        "FST": 5,
        "LO1": 5,
        "LO2": 5,
        "LO3": 5,
        "MT": 2,
        "MST": 3,
        "PIT": 9,
        "IPS1": 9,
        "RSC": 9,
        "FFC": 9,
        # Add other ROIs as needed
    }

    LOGGER.info("Step 3: Classified streams based on ROI names")

    sorted_models = sorted(df["model"].unique())

    model_to_x = {model: i for i, model in enumerate(sorted_models)}

    # Step 2: Color map for ROI names
    roi_names = df["roi"].unique()
    if split != "both":
        roi_palette = sns.color_palette("hls", 11)  # or another palette
        roi_color_map = {
            roi: roi_palette[roi_hierarchy_map.get(roi, 10)] for roi in roi_names
        }
    else:
        train_palette = sns.color_palette("Reds", 11)
        test_palette = sns.color_palette("Blues", 11)

        train_roi_color_map = {
            roi: train_palette[roi_hierarchy_map.get(roi, 10)] for roi in roi_names
        }
        test_roi_color_map = {
            roi: test_palette[roi_hierarchy_map.get(roi, 10)] for roi in roi_names
        }
        roi_color_map = train_roi_color_map

    LOGGER.info("Step 2: Created ROI color map")

    # Step 3: Marker map for ROI stream
    roi_stream_marker = {
        "dorsal": "o",  # circle
        "ventral": "s",  # square
        "early": "d",
        "unknown": "x",
    }

    LOGGER.info("Step 3: Created ROI stream marker map")

    # horizontal constant offset to right/left for ventral/dorsal

    offset_map = {"early": -0.3, "ventral": -0.075, "dorsal": 0.075, "unknown": 0.3}

    # Filter for the score split you care about
    if split != "both":
        df = df[df["split"] == split]
    # Step 4: Create plot
    fig, ax = plt.subplots(figsize=(20, 6))

    # Plot each row with correct color and marker
    for _, row in df.iterrows():
        x_val = model_to_x[row["model"]] + offset_map[row["stream"]]
        y_val = row["score"]
        if split != "both":
            color = roi_color_map.get(row["roi"], "gray")
        else:
            if row["split"] == "test":
                color = test_roi_color_map.get(row["roi"], "gray")
            elif row["split"] == "train":
                color = train_roi_color_map.get(row["roi"], "gray")
            else:
                color = "gray"
        marker = roi_stream_marker.get(row["stream"], "x")  # fallback marker

        ax.scatter(
            x_val,
            y_val,
            color=color,
            marker=marker,
            edgecolor=None,
            linewidth=0.3,
            s=50,
            alpha=0.8,
        )

    grouped_df = df.groupby(
        ["stream", "model", "split"],
        as_index=False,
    )["score"].mean()

    for _, row in grouped_df.iterrows():
        x_val = model_to_x[row["model"]] + offset_map[row["stream"]]
        y_val = row["score"]
        # color = "k"
        facecolor = "red" if row["split"] == "train" else "blue"
        marker = roi_stream_marker.get(row["stream"], "x")  # fallback marker

        ax.scatter(
            x_val,
            y_val,
            # color=color,
            marker=marker,
            facecolor=facecolor,
            linewidth=0.3,
            s=50,
            alpha=0.8,
        )

    # Set x-axis ticks and labels
    ax.set_xticks(list(model_to_x.values()))
    ax.set_xticklabels(sorted_models, rotation=90)

    ax.set_xlabel("Model (alphabetical)")
    ax.set_ylabel("Alignment Score")
    if split != "both":
        ax.set_title(f"Alignment by Model, ROI Name, and Stream (split: {split})")
    else:
        ax.set_title(
            "Alignment by Model, ROI Name, "
            "and Stream (Train (red) vs. Test (blue) splits)",
        )

    LOGGER.info("Step 4: Plotted scores for each ROI and model")

    # Step 5: Create unified legend, two columns: ventral and dorsal

    # Sort ROIs by hierarchy
    sorted_rois = sorted(roi_hierarchy_map.items(), key=lambda x: x[1])
    sorted_roi_names = [roi for roi, _ in sorted_rois if roi in roi_color_map]

    # Split into ventral and dorsal legend handles
    ventral_handles = []
    dorsal_handles = []
    early_handles = []
    unknown_handles = []
    if split == "both":
        ventral_handles_test = []
        dorsal_handles_test = []
        early_handles_test = []
        unknown_handles_test = []

    for roi in sorted_roi_names:
        stream = classify_stream(roi)
        marker = roi_stream_marker[stream]
        color = roi_color_map[roi]
        if split != "both":
            color_train = roi_color_map[roi]
        # Train handle
        else:
            color_train = train_roi_color_map[roi]
        handle_train = Line2D(
            [0],
            [0],
            marker=marker,
            color="w",
            label=roi,
            markerfacecolor=color_train,
            markeredgecolor="k",
            markersize=8,
        )
        if split == "both":
            # Test handle
            color_test = test_roi_color_map[roi]
            handle_test = Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                label=roi,
                markerfacecolor=color_test,
                markeredgecolor="k",
                markersize=8,
            )

        if stream == "ventral":
            ventral_handles.append(handle_train)
            if split == "both":
                ventral_handles_test.append(handle_test)
        elif stream == "dorsal":
            dorsal_handles.append(handle_train)
            if split == "both":
                dorsal_handles_test.append(handle_test)
        elif stream == "early":
            early_handles.append(handle_train)
            if split == "both":
                early_handles_test.append(handle_test)
        elif stream == "unknown":
            unknown_handles.append(handle_train)
            if split == "both":
                unknown_handles_test.append(handle_test)

    # Pad shorter list with invisible handles to align columns
    max_len = max(
        len(ventral_handles),
        len(dorsal_handles),
        len(early_handles),
        len(unknown_handles),
    )
    if split == "both":
        max_len = max(
            max_len,
            max(
                len(ventral_handles_test),
                len(dorsal_handles_test),
                len(early_handles_test),
                len(unknown_handles_test),
            ),
        )

    def pad_handles(handles, length):
        while len(handles) < length:
            handles.append(Line2D([0], [0], color="none", label=""))

    pad_handles(ventral_handles, max_len)
    pad_handles(dorsal_handles, max_len)
    pad_handles(early_handles, max_len)
    pad_handles(unknown_handles, max_len)
    if split == "both":
        pad_handles(ventral_handles_test, max_len)
        pad_handles(dorsal_handles_test, max_len)
        pad_handles(early_handles_test, max_len)
        pad_handles(unknown_handles_test, max_len)

    combined_handles = (
        ventral_handles + dorsal_handles + early_handles + unknown_handles
    )
    if split == "both":
        combined_handles = combined_handles + (
            ventral_handles_test
            + dorsal_handles_test
            + early_handles_test
            + unknown_handles_test
        )

    # Plot legend
    title = "Ventral (■), Dorsal (●), Early (d), Unknown (x)"
    if split == "both":
        title = (
            "Ventral (■), Dorsal (●), Early (d), Unknown (x)\n"
            "Train (Reds), Test (Blues)"
        )
    ax.legend(
        handles=combined_handles,
        labels=[h.get_label() for h in combined_handles],
        title=title,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        ncol=6,
        columnspacing=1.5,
        handletextpad=0.5,
    )

    LOGGER.info("Step 5: Created unified legend for ROIs and streams")

    if results_subdir:
        results_dir = results_dir / results_subdir
        results_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(results_dir)

    if split != "both":
        plot_fname_final = "_".join([plot_fname, split])
    else:
        plot_fname_final = plot_fname
    save_this(
        results_dir,
        plot_fname_final,
        formats=["png"],
        fig=plt.gcf(),
        transparent=False,
    )

    LOGGER.info(f"Saved plot to {results_dir}")


def plot_scoresheet(scoresheet, config):
    """Given a scoresheet, plots the scoresheet."""
    config_str = config["config_str"]

    #  Convert scoresheet to a pandas DataFrame
    df = pd.DataFrame(scoresheet)

    df = pd.DataFrame.transpose(df)
    df = df.reset_index()
    df = df[df["score_set"] == "test"]

    heatmap_data = df.drop(
        columns=[
            "score_set",
            "train_type",
            "distance_1",
            "distance_2",
            "model_layer_index",
            "model_layer",
            "region",
        ]
    )

    heatmap_data = heatmap_data.pivot(index="index", columns="model", values="score")
    heatmap_data = heatmap_data.astype(float)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    sns.heatmap(
        heatmap_data,
        annot=True,  # Annotate cells with the score values
        fmt=".4f",  # Format the annotations to 4 decimal places
        cmap="coolwarm",  # Use a diverging colormap
        cbar_kws={"label": "Score"},  # Add a label to the color bar
    )

    # Add labels and title
    plt.title(f"Scoresheet Heatmap: {config_str}")
    plt.xlabel("Model")
    plt.ylabel("Region")

    # Save the heatmap to a file
    output_path = Path(config["output_dir"]) / "scoresheet_heatmap"
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / f"scoresheet_heatmap_{config_str}.png"
    # if there is no file with the same name, save it
    if not output_path.exists() or config["overwrite"]:
        plt.savefig(output_path)
    LOGGER.info(f"Heatmap saved to {output_path}")

    # Show the heatmap
    plt.show()


def plot_rdm(
    rdm: npt.NDArray[np.float64],
    title: str,
    output_path: Path,
    num_samples: int = 100,
) -> None:
    """Plots a RDM as a heatmap.

    Parameters:
        rdm: The RDM as numpy array of shape (n_samples, n_samples)
        title: The title of the plot
        output_path: Output file path
        num_samples: The number of samples to plot
    """
    rdm = rdm[:num_samples, :num_samples]

    plt.figure(figsize=(5, 5))
    sns.heatmap(rdm, cmap="crest", annot=False, cbar=True)
    plt.title(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


@click.command()
@click.option(
    "--output-supdir",
    "output_supdir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--output-dirs",
    "output_dirs",
    multiple=True,
    default=[],
)
@click.option("--filename", "filename", type=str, required=True)
@click.option(
    "--results-dir",
    "results_dir",
    type=click.Path(exists=False, file_okay=False, path_type=Path),
    required=True,
)
@click.option("--results-subdir", "results_subdir", type=str, default="")
@click.option("--plot-fname", "plot_fname", type=str, default="latest_results")
@click.option(
    "--split",
    "split",
    type=str,
    required=True,
    help="one of: 'train', 'test', or 'both'",
)
@click.option("--procrustes/--no-procrustes", "procrustes", type=bool, default=False)
@click.option("--format", "res_format", type=str, default="new")
def plot_layers_brain_model(
    output_supdir: Path,
    output_dirs: list[str],
    filename: str,
    results_dir: Path,
    results_subdir: str,
    split: str,
    plot_fname: str = "matching_layers",
    procrustes: bool = False,
    res_format: str = "new",
):
    """Plot the alignment of layers of different models vs. ROIs.

    Given a list of output directories, pulls in the results from scoresheets stored
    in these directories and creates a plot alignment per ROI vs. models.
    """
    line_plot = True
    line_plot_per_model = False

    if len(output_dirs) == 0:
        output_dirs = os.listdir(output_supdir)
    all_scoresheets = []
    n = 0
    # filename = "scoresheet.pkl" if not procrustes else "scoresheet_procrustes.pkl"

    for output_dir in output_dirs:
        if not os.path.isdir(os.path.join(output_supdir, output_dir)):
            continue

        if res_format == "old":  # supporting pkl files
            fpath = Path(
                os.path.join(output_supdir, output_dir, "scoresheets", filename)
            )
            if fpath.exists():
                with open(fpath, "rb") as f:
                    scoresheet = pickle.load(f)
                    all_scoresheets.extend(scoresheet)
                    n += 1
            else:
                f"Couldn't load scoresheet for run {output_dir}"

        elif res_format == "new":
            LOGGER.info(f"Loading scoresheet for run {output_dir}")
            os.listdir(os.path.join(output_supdir, output_dir))
            fpath = Path(
                os.path.join(
                    output_supdir,
                    output_dir,
                    "scoresheets",
                    filename,
                )
            )
            if fpath.exists():
                scoresheet = pd.read_csv(fpath)
                all_scoresheets.append(scoresheet)
                n += 1
            else:
                LOGGER.info(f"Couldn't load scoresheet for run {output_dir}")

    if len(all_scoresheets) >= 1 and res_format == "old":
        # Step 1: Load the list of dicts into a DataFrame
        df = pd.DataFrame(all_scoresheets)
        LOGGER.info("Step 1: Loaded scoresheets into DataFrame")
        LOGGER.info(f"Loaded {len(df)} rows from {n} runs.")

        # Step 2: Rename fields for clarity (optional)
        df = df.rename(
            columns={
                "region": "roi",
                "train_type": "metric",
                "score_set": "split",
                "model_str": "model",
                "layer_str": "layer",
            }
        )

        LOGGER.info("Step 2: Renamed columns for clarity")

    elif len(all_scoresheets) >= 1 and res_format == "new":
        df = pd.concat(all_scoresheets)
    else:
        return

    # for every model, rename the layer to be a number from 0 to 10
    # this is because the layer names are not consistent across models

    for model in df["model"].unique():
        # make a dict out of the layer names but keep their order
        layer_names = df[df["model"] == model]["layer"].unique()
        layer_map = {name: i for i, name in enumerate(layer_names)}
        df.loc[df["model"] == model, "layer"] = df.loc[
            df["model"] == model, "layer"
        ].map(layer_map)

    # now i want to write code that makes one plot per roi
    # on the x axis i want the models, sorted by mean score across all layers
    # on the y axis i want the layer score
    # its supposed to be a scatter plot
    # the layers are color coded from 0 to 10

    # save the plots to output_dir/plot_fname and then roi.png
    # if results_subdir is specified, save to results_dir/results_subdir/plot_fname
    # if results_subdir is not specified, save to results_dir/plot_fname

    # lets only do test split for now
    if split != "both":
        df = df[df["split"] == split]

        if line_plot:
            if not line_plot_per_model:
                # make a line plot that displays each layer as a separate line
                # for each roi, plot the layer scores for each model
                for roi in df["roi"].unique():
                    roi_df = df[df["roi"] == roi]

                    # sort models by mean score across all layers
                    model_scores = roi_df.groupby("model")["score"].max()
                    model_scores = model_scores.sort_values(ascending=True)
                    sorted_models = model_scores.index.tolist()

                    # Set the categorical order for 'model' so seaborn respects the
                    # sorting
                    roi_df = roi_df.copy()
                    roi_df["model"] = pd.Categorical(
                        roi_df["model"],
                        categories=sorted_models,
                        ordered=True,
                    )

                    plt.figure(figsize=(10, 6))
                    ax = plt.gca()
                    # Dynamically generate a palette matching the number of unique
                    # layers
                    n_layers = roi_df["layer"].nunique()
                    dynamic_palette = sns.color_palette("viridis", n_layers)
                    # Ensure all layers are shown in legend
                    layers_sorted = sorted(roi_df["layer"].unique())
                    for i, layer in enumerate(layers_sorted):
                        layer_df = roi_df[roi_df["layer"] == layer]
                        sns.lineplot(
                            data=layer_df,
                            x="model",
                            y="score",
                            label=str(layer),
                            color=dynamic_palette[i],
                            marker="o",
                            markersize=5,
                            linewidth=2,
                            ax=ax,
                        )
                    plt.title(f"Layer Scores for {roi} ({split})")
                    plt.xlabel("Model")
                    plt.ylabel("Score")
                    plt.xticks(rotation=90)
                    # manually create legend handles to match dynamic palette
                    handles = [
                        Line2D(
                            [0],
                            [0],
                            color=dynamic_palette[i],
                            marker="o",
                            linestyle="-",
                            markersize=5,
                            linewidth=2,
                        )
                        for i in range(len(layers_sorted))
                    ]
                    labels = [str(layer) for layer in layers_sorted]
                    plt.legend(
                        handles=handles,
                        labels=labels,
                        title="Layers",
                        bbox_to_anchor=(1.05, 1),
                        loc="upper left",
                    )

                    output_path = results_dir / (
                        results_subdir if results_subdir else ""
                    )
                    output_path.mkdir(parents=True, exist_ok=True)
                    output_file = output_path / f"{plot_fname}_{roi}.png"

                    plt.savefig(output_file, bbox_inches="tight")
                    plt.close()

                    LOGGER.info(f"Saved plot for {roi} to {output_file}")
            elif line_plot_per_model:
                """Was ich meinte war, eine Kurve pro Modell und ROI
                (also ein Plot pro ROI, so wie jetzt, aber darin eine Kurve pro Modell)
                und als x-achse die layer depth, und als y-achse die similarity zu dem
                ROI"""
                # make a line plot that displays each model as a separate line
                # for each roi, plot the layer scores for each model
                # this is similar to the above, but we want to plot each model as a
                # separate line
                # so we will iterate over each model and plot the layer scores for that
                # model
                # make sure alpha is not 1

                for roi in df["roi"].unique():
                    roi_df = df[df["roi"] == roi]

                    # make the layer a categorical variable (ensure all expected layers
                    # are present)
                    roi_df = roi_df.copy()
                    expected_layers = list(range(11))
                    roi_df["layer"] = pd.Categorical(
                        roi_df["layer"],
                        categories=expected_layers,
                        ordered=True,
                    )

                    plt.figure(figsize=(10, 6))
                    ax = plt.gca()
                    # Boxplot: x=layer (categorical), y=score, hue=model
                    sns.boxplot(
                        data=roi_df,
                        x="layer",
                        y="score",
                        ax=ax,
                        showfliers=False,
                    )
                    plt.title(f"Layer Scores for {roi} ({split})")
                    plt.xlabel("Layer")
                    plt.ylabel("Score")
                    plt.xticks(rotation=90)

                    output_path = results_dir / (
                        results_subdir if results_subdir else ""
                    )
                    output_path.mkdir(parents=True, exist_ok=True)
                    output_file = output_path / f"{plot_fname}_{roi}.png"

                    plt.savefig(output_file, bbox_inches="tight")
                    plt.close()

                    LOGGER.info(f"Saved plot for {roi} to {output_file}")

    else:
        for roi in df["roi"].unique():
            roi_df = df[df["roi"] == roi]

            # sort models by mean score across all layers
            model_scores = roi_df.groupby("model")["score"].max()
            model_scores = model_scores.sort_values(ascending=True)
            sorted_models = model_scores.index.tolist()

            # Set the categorical order for 'model' so seaborn respects the sorting
            roi_df = roi_df.copy()
            roi_df["model"] = pd.Categorical(
                roi_df["model"],
                categories=sorted_models,
                ordered=True,
            )

            plt.figure(figsize=(10, 6))
            # Dynamically generate a palette matching the number of unique layers
            n_layers = roi_df["layer"].nunique()
            dynamic_palette = sns.color_palette("viridis", n_layers)
            sns.scatterplot(
                data=roi_df,
                x="model",
                y="score",
                hue="layer",
                palette=dynamic_palette,
                s=100,
            )

            plt.title(f"Layer Scores for {roi} ({split})")
            plt.xlabel("Model")
            plt.ylabel("Score")
            plt.xticks(rotation=90)
            plt.legend(title="Layers", bbox_to_anchor=(1.05, 1), loc="upper left")

            output_path = results_dir / (results_subdir if results_subdir else "")
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / f"{plot_fname}_{roi}.png"

            plt.savefig(output_file, bbox_inches="tight")
            plt.close()

            LOGGER.info(f"Saved plot for {roi} to {output_file}")


if __name__ == "__main__":
    plot_layers_brain_model()
    # plot_results()
