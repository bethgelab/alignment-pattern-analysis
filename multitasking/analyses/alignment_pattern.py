import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap, rgb_to_hsv
from typing import Dict, List, Tuple

metric_palette = ["darkblue", "darkred"]
metric_color_dict = dict(
    zip(["linear_predictivity", "rsa"], metric_palette, strict=False)
)


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import colorsys
from typing import Dict, List, Tuple


def generate_grouped_cubehelix_colors(
    groups: Dict[str, List[str]],
    base_cmap_name: str = "tab20",
    rot: float = -0.75,
    light: float = 0.85,
    dark: float = 0.25,
    gamma: float = 1.0,
    sat_scale: float = 1.2,
) -> Tuple[
    Dict[Tuple[str, str], Tuple[float, float, float, float]],
    Dict[str, ListedColormap],
]:
    item_color_dict: Dict[Tuple[str, str], Tuple[float, float, float, float]] = {}
    group_cmap_dict: Dict[str, ListedColormap] = {}

    group_names = list(groups.keys())
    n_groups = len(group_names)

    base_cmap = plt.get_cmap(base_cmap_name, n_groups)
    base_colors = [base_cmap(i)[:3] for i in range(n_groups)]

    for gi, group_name in enumerate(group_names):
        items = groups[group_name]
        n_items = len(items)

        # Cubehelix luminance ramp
        ramp = sns.cubehelix_palette(
            n_colors=n_items,
            start=0.5,
            rot=rot,
            gamma=gamma,
            light=light,
            dark=dark,
            reverse=False,
            as_cmap=False,
        )

        base_rgb = base_colors[gi]

        # Apply base hue, preserve ramp lightness
        tinted = []
        base_h, _, base_s = colorsys.rgb_to_hls(*base_rgb)

        for rgb in ramp:
            _, l, _ = colorsys.rgb_to_hls(*rgb)
            s = min(1.0, base_s * sat_scale)
            tinted.append(colorsys.hls_to_rgb(base_h, l, s))

        rgba_palette = [(*rgb, 1.0) for rgb in tinted]

        for ii, item in enumerate(items):
            item_color_dict[item] = rgba_palette[ii]

        group_cmap_dict[group_name] = ListedColormap(rgba_palette)

    return item_color_dict, group_cmap_dict



def find_max_aligned_roi(
    alignment_pattern_brain: dict[str, dict[str, dict[str, float]]] \
        | dict[str, dict[str, dict[str, dict[str, float]]]],
    metric: str,
    roi: str,
    valid_rois: list[str],
    subjects: None | list[str] = None,
) -> tuple[str, float]:
    """Find the ROI with the maximum alignment score for a given metric and ROI."""
    max_aligned_roi = ""
    max_alignment_score: float = 0.0
    if subjects is None:
        for _roi in valid_rois:
            score = alignment_pattern_brain[metric][roi][_roi]
            if score > max_alignment_score:
                max_alignment_score = score
                max_aligned_roi = _roi
    else:
        for _roi in valid_rois:
            score = float(
                np.mean([alignment_pattern_brain[metric][roi][subject][_roi]
                            for subject in subjects])
            )
            if score > max_alignment_score:
                max_alignment_score = score
                max_aligned_roi = _roi
    return max_aligned_roi, max_alignment_score


def get_roi_colormap(
    rois: list[str], method: str = "tab20"
) -> tuple[dict[str, tuple[float, ...]], ListedColormap]:
    """Create a colormap for ROIs.

    Args:
        rois: List of ROI names
        method: Color method to use. Options:
            - 'tab20': Use tab20 colormap (default, good for many distinct colors)
            - 'stream': Group by visual stream (early/ventral/dorsal)
            - 'cubehelix_stream': Like 'stream' but uses three distinct
              cubehelix palettes (4–8 colors each) for better distinguishability
            - 'xkcd_stream': Stream-based coloring using xkcd color palette with
              shades of blue (early), red (ventral), and green (dorsal)

    Returns:
        tuple of (roi_color_dict, cmap) where:
        - roi_color_dict: dictionary mapping ROI names to RGBA tuples
        - cmap: ListedColormap object that can be used to add a colorbar
    """
    stream_rois = {
            "early": ["V1", "V2", "V3"],
            "ventral": ["V4", "V8", "PIT", "FFC"],
            "dorsal": ["V3A", "V3B", "V6", "V6A", "V7", "IPS1",
                       "LO1", "LO2", "LO3", "MT", "MST", "FST"],
        }

    if method == "tab20":
        # Get colors from tab20 colormap
        colors = plt.cm.tab20(range(len(rois)))  # type: ignore[attr-defined]
    elif method == "stream":
        # Stream-based coloring
        # Define stream assignments
        stream_cmaps = [
            plt.cm.Blues,  # type: ignore[attr-defined]
            plt.cm.Reds,  # type: ignore[attr-defined]
            plt.cm.Purples,  # type: ignore[attr-defined]
        ]

        colors = []
        for roi in rois:
            # Find which stream this ROI belongs to
            found = False
            for stream_idx, (_stream_name, stream_list) in enumerate(
                stream_rois.items()
            ):
                if roi in stream_list:
                    idx_in_stream = stream_list.index(roi)
                    num_in_stream = len(stream_list)
                    # Map within stream using appropriate intensity
                    intensity = 0.3 + 0.4 * (idx_in_stream / max(1, num_in_stream - 1))
                    color = stream_cmaps[stream_idx](intensity)
                    colors.append(color)
                    found = True
                    break
            if not found:
                # Fallback to gray if ROI not found in any stream
                colors.append((0.5, 0.5, 0.5, 1.0))
        colors = np.array(colors)
    elif method == "cubehelix_stream":
        # Stream-based coloring using three distinct cubehelix palettes

        # Reuse general generator
        item_colors, group_cmaps = generate_grouped_cubehelix_colors(
            stream_rois,
            starts=(0.5, 2.0, 4.0),
            rots=(-0.75, 0.75, 1.0),
            hue=1.0,
            light=0.85,
            dark=0.25,
        )
        colors = []
        for roi in rois:
            found = False
            for stream_name, stream_list in stream_rois.items():
                if roi in stream_list:
                    colors.append(
                        item_colors.get((stream_name, roi), (0.5, 0.5, 0.5, 1.0))
                    )
                    found = True
                    break
            if not found:
                colors.append((0.5, 0.5, 0.5, 1.0))
        colors = np.array(colors)
    elif method == "xkcd_stream":
        # Stream-based coloring using xkcd colors with consistent hue per stream

        # Dynamically select xkcd colors near a target hue to avoid invalid names
        # target hues in degrees: blue ~ 240, red ~ 0, green ~ 120
        target_hues = {"early": 240.0, "ventral": 0.0, "dorsal": 120.0}

        def circular_hue_distance(h1: float, h2: float) -> float:
            d = abs(h1 - h2) % 360.0
            return min(d, 360.0 - d)

        def get_xkcd_shades(
            target_hue_deg: float, n_needed: int
        ) -> list[tuple[float, float, float]]:
            candidates: list[tuple[float, float, float, float, str]] = []
            for name, hex_color in sns.xkcd_rgb.items():
                rgb = mcolors.to_rgb(hex_color)
                hsv = mcolors.rgb_to_hsv(rgb)
                hue_deg = hsv[0] * 360.0
                sat = hsv[1]
                val = hsv[2]
                # Filter: close hue, sufficiently saturated for distinctness
                if (
                    circular_hue_distance(hue_deg, target_hue_deg) <= 35.0
                    and sat >= 0.35
                ):
                    candidates.append((hue_deg, sat, val, 0.0, name))

            # Remove very dark colors by preferring higher value (brightness)
            def filter_by_value(cands, min_v):
                return [c for c in cands if c[2] >= min_v]

            # Try progressively relaxed value thresholds
            filtered = filter_by_value(candidates, 0.55)
            if not filtered:
                filtered = filter_by_value(candidates, 0.45)
            if not filtered:
                filtered = candidates

            # Sort by value (lightness) then sample evenly to cover the range
            filtered.sort(key=lambda t: t[2])
            if not filtered:
                # Fallback: if no candidates, use generic colormap samples
                base = plt.cm.hsv(target_hue_deg / 360.0)  # type: ignore[attr-defined]
                return [base[:3]] * n_needed
            # Pick evenly across sorted list
            idxs = (
                np.linspace(0, len(filtered) - 1, max(1, n_needed)).round().astype(int)
            )
            picked_hex = [sns.xkcd_rgb[filtered[i][4]] for i in idxs]
            return [mcolors.to_rgb(h) for h in picked_hex]

        # Build per-stream palettes sized to the number of ROIs in each stream
        stream_palettes: dict[str, list[tuple[float, float, float]]] = {}
        for stream_name, stream_list in stream_rois.items():
            n_stream = len(stream_list)
            pal_rgb = get_xkcd_shades(target_hues[stream_name], n_stream)
            stream_palettes[stream_name] = pal_rgb

        colors = []
        for roi in rois:
            found = False
            for stream_name, stream_list in stream_rois.items():
                if roi in stream_list:
                    idx = stream_list.index(roi)
                    rgb = stream_palettes[stream_name][idx]
                    colors.append((*rgb, 1.0))
                    found = True
                    break
            if not found:
                colors.append((0.5, 0.5, 0.5, 1.0))
        colors = np.array(colors)
    else:
        raise ValueError(
            f"Unknown method: {method}. Choose 'tab20', 'stream', "
            f"'cubehelix_stream', or 'xkcd_stream'."
        )

    # Create dictionary mapping ROIs to colors
    roi_color_dict = {
        roi: tuple(color) for roi, color in zip(rois, colors, strict=False)
    }

    # Create a ListedColormap for the colorbar
    cmap = ListedColormap(colors)

    return roi_color_dict, cmap


def build_graph(
    connectivity_dict: dict[str, tuple[tuple[str, ...], tuple[str, ...], int]],
) -> nx.DiGraph:
    """Build a graph from the connectivity dictionary."""
    graph = nx.DiGraph()
    for region, (inputs, outputs, level) in connectivity_dict.items():
        graph.add_node(region, level=level)
        for target in outputs:
            graph.add_edge(region, target)
        for source in inputs:
            graph.add_edge(source, region)
    return graph


def prune_graph(graph: nx.DiGraph, **kwargs) -> tuple[nx.DiGraph, list[str]]:
    """Prune the graph."""
    graph = graph.subgraph(kwargs["nodes_to_keep"]).copy()
    valid_nodes = list(graph.nodes)
    return graph, valid_nodes


def build_strength_matrix(
    graph: nx.DiGraph, rois, decay: float = 0.9
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build a strength matrix from the graph."""
    assert not (graph.is_directed()), "Graph must be undirected"
    strength_matrix = np.zeros((len(rois), len(rois)))
    for i, node1 in enumerate(rois):
        for j, node2 in enumerate(rois):
            if i == j:
                strength_matrix[i, j] = 1.0
            else:
                try:
                    path_len = nx.shortest_path_length(
                        graph, node1, node2, weight="weight"
                    )
                    strength = decay**path_len
                    strength_matrix[i, j] = strength
                    # strength_matrix[j, i] = strength
                except nx.NetworkXNoPath:
                    pass
    strength_df = pd.DataFrame(strength_matrix, index=rois, columns=rois)
    return strength_matrix, strength_df


# def get_brain_brain_alignment_pattern_per_subject(
#     intersubject_df: pd.DataFrame,
#     valid_rois: list[str],
#     metric: str,
#     subjects: list[str],
#     split: str = "test",
#     **kwargs,
# ) -> dict[str, dict[str, list[float]]]:
#     """Get the brain-brain alignment pattern."""
#     alignment_patterns: dict[str, dict[str, dict[str, list[float]]]] = {}
#     for roi in valid_rois:
#         alignment_patterns[roi] = {subject: {} for subject in subjects}
#         # subset to the current roi
#         roi_df = intersubject_df[
#             (intersubject_df["layer"] == roi)
#             & (intersubject_df["metric"] == metric)
#             & (intersubject_df["split"] == split)
#         ].copy()
#         # group by "layer" and get the mean score. In intersubject, we compare
#         # ROIs to ROIs, and we call the predicting ROIs "layers" and
#         # the predicted ROIs "ROIs"
#         for subject in subjects:
#             ap = roi_df.query(f"subject == '{subject}'")\
#                 [['score', 'roi']].set_index('roi').to_dict()
#             alignment_patterns[roi][subject] = {_roi: ap['score'][_roi]\
#                 for _roi in valid_rois}
#     return alignment_patterns


def get_within_roi_aps(alignment_pattern_brain, roi, metric, subjects, valid_rois):
    within_roi_aps = np.zeros(len(subjects))
    alignment_patterns = np.asarray([
        [alignment_pattern_brain[metric][roi][subject][_roi] for _roi in valid_rois]
        for subject in subjects
    ])
    for i, _subject in enumerate(subjects):
        avg = (alignment_patterns.sum(axis=0) - alignment_patterns[i]) /\
            (alignment_patterns.shape[0] - 1)
        within_roi_aps[i] = np.corrcoef(avg, alignment_patterns[i])[0, 1]
    return within_roi_aps


def get_brain_brain_alignment_pattern(
    intersubject_df: pd.DataFrame,
    valid_rois: list[str],
    metric: str,
    split: str = "test",
    **kwargs
) -> dict[str, dict[str, float]]:
    """Get the brain-brain alignment pattern."""
    alignment_pattern: dict[str, dict[str, float]] = {}
    for roi in valid_rois:
        # subset to the current roi
        roi_df = intersubject_df[(intersubject_df["roi"]==roi) &
                                (intersubject_df['metric']==metric) &
                                (intersubject_df['split']==split)].copy()
        # group by "layer" and get the mean score. In intersubject, we compare
        # ROIs to ROIs, and we call the predicting ROIs "layers" and
        # the predicted ROIs "ROIs"
        roi_dict = (
            roi_df.groupby("layer", as_index=False).agg(
                count=("score", "count"),
                score_mean=("score", "mean"),
                score_std=("score", "std"),
            )[["layer", "score_mean"]]
            .set_index("layer")["score_mean"]
            .to_dict()
        )
        # add the alignment pattern to the dictionary
        alignment_pattern[roi] = {layer: float(roi_dict[layer]) for layer in valid_rois}
    return alignment_pattern



def get_model_brain_alignment_pattern(
    full_df: pd.DataFrame,
    model: str,
    layer: str,
    valid_rois: list[str],
    metric: str,
    split: str = "test",
    **kwargs,
) -> dict[str, float]:
    """Get the model-brain alignment pattern."""
    # subset to the current metric, split, model, and layer
    model_layer_df = full_df[
        (full_df["metric"] == metric)
        & (full_df["split"] == split)
        & (full_df["model"] == model)
        & (full_df["layer"] == layer)
    ].copy()
    roi_df = (
        model_layer_df.groupby("roi", as_index=False)
        .agg(
            count=("score", "count"),
            score_mean=("score", "mean"),
            score_std=("score", "std"),
        )[["roi", "score_mean"]]
        .set_index("roi")["score_mean"]
        .to_dict()
    )
    # add the alignment pattern to the dictionary
    alignment_pattern_list = {_roi: roi_df[_roi] for _roi in valid_rois}
    return alignment_pattern_list


def plot_graph(graph: nx.DiGraph, **kwargs) -> None:
    """Plot the graph."""
    plt.figure(figsize=(10, 10))
    nx.draw(graph, **kwargs)
    plt.show()


def plot_alignment_pattern(
    alignment_pattern_1: dict[str, dict[str, dict[str, float]]],
    alignment_pattern_1_type: str,
    alignment_pattern_2: dict[str, dict[str, dict[str, dict[str, float]]]],
    alignment_pattern_2_type: str,
    aps: dict[str, dict[str, dict[str, float]]],
    valid_rois: list[str],
    metrics: list[str],
    roi: str,
    aps_cmap_str: str = "RdBu_r",
    roi_color_dict: dict[str, tuple[float, float, float]] | None = None,
    **plot_kwargs,
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Plot alignment patterns."""
    fig, [ax1, axcm] = plt.subplots(
        1, 2, figsize=(4, 2), gridspec_kw={"width_ratios": [5, 1], "wspace": 0.5}
    )
    # Create twin axis for second pattern
    ax2 = ax1.twinx()

    for metric in metrics:
        if alignment_pattern_1_type == "brain":
            # Color can be either string (from metric_color_dict) or tuple
            # (from roi_color_dict)
            color: str | tuple[float, float, float]
            if roi_color_dict is not None:
                color = roi_color_dict[roi]
            else:
                color = metric_color_dict[metric]
            y = [alignment_pattern_1[metric][roi][_roi] for _roi in valid_rois]
            ax1.plot(
                range(len(valid_rois)),
                y,
                color=color,  # type: ignore[arg-type]
                linestyle="--",
                label=f"{roi} ap",
                **plot_kwargs,
            )

        if alignment_pattern_2_type == "model":
            n_predictors = len(alignment_pattern_2[metric][roi].keys())
            predictors = list(alignment_pattern_2[metric][roi].keys())
            palette = sns.color_palette(aps_cmap_str, n_predictors)
            cmap = ListedColormap(palette)
            sm = ScalarMappable(
                cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=n_predictors)
            )
            sorted_aps_values = np.sort(list(aps[metric][roi].values()))
            sorting_idxs = np.argsort(list(aps[metric][roi].values()))
            sorted_predictors = [predictors[idx] for idx in sorting_idxs]
            for predictor in sorted_predictors:
                ordinal_value = np.where(
                    sorted_aps_values == aps[metric][roi][predictor]
                )[0][0]
                y = [
                    alignment_pattern_2[metric][roi][predictor][_roi]
                    for _roi in valid_rois
                ]
                ax2.plot(
                    range(len(valid_rois)),
                    y,
                    color=sm.to_rgba(np.array([ordinal_value]))[0],  # type: ignore[arg-type]
                    #  linestyle='--',
                    #  label=f"{short_predictor}, {aps[metric][roi][predictor]:.2f}",
                    **plot_kwargs,
                )
    # Configure twin axis
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.spines["right"].set_visible(True)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    from matplotlib.ticker import MaxNLocator

    ax1.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax1.set_xticks(range(len(valid_rois)))
    ax1.set_xticklabels(valid_rois, rotation=90)
    ax1.set_xlabel("ROIs")
    ax1.set_ylabel("alignment score")
    # ax2.set_ylabel("model-brain alignment score")
    # Set slight grey background
    # ax1.set_facecolor('#f5f5f5')  # Light grey background
    # ax2.set_facecolor('#f5f5f5')   # Match twin axis background

    ax1.set_title(f"{roi}, {metric}")
    plt.colorbar(sm, cax=axcm)
    axcm.set_yticks(
        [i + 0.5 for i in range(len(alignment_pattern_2[metric][roi].keys()))]
    )

    axcm.set_yticklabels([predictor[:40] for predictor in sorted_predictors])
    # axcm.yaxis.set_label_position("left")
    # axcm.set_ylabel("aps score", )
    axcm.set_title("aps score")
    n_predictors = len(predictors)
    for predictor in predictors:
        ordinal_value = np.where(sorted_aps_values == aps[metric][roi][predictor])[0][0]
        # Find center of each color segment in axis coordinates
        y = (ordinal_value + 0.5) / (n_predictors)
        axcm.text(
            0.5,
            y,
            f"{aps[metric][roi][predictor]:.2f}",
            ha="center",
            va="center",
            color="black",
            fontsize=10,
            transform=axcm.transAxes,
        )
    ax1.legend()

    return fig, ax1, ax2




def plot_alignment_pattern_scores(
    alignment_pattern_1: dict[str, dict[str, dict[str, float]]],
    alignment_pattern_1_type: str,
    alignment_pattern_2: dict[str, dict[str, dict[str, dict[str, float]]]],
    alignment_pattern_2_type: str,
    aps: dict[str, dict[str, dict[str, float]]],
    valid_rois: list[str],
    metrics: list[str],
    roi: str,
    aps_cmap_str: str = "RdBu_r",
    roi_color_dict: dict[str, tuple[float, float, float]] | None = None,
    **plot_kwargs,
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Plot alignment patterns."""
    fig, [ax1, ax2, axcm] = plt.subplots(
        1, 3, figsize=(8, 2), gridspec_kw={"width_ratios": [5, 5, 1], "wspace": 0.7}
    )
    # Create twin axis for second pattern
    axtwinx = ax1.twinx()

    for metric in metrics:
        if alignment_pattern_1_type == "brain":
            # Color can be either string (from metric_color_dict) or tuple
            # (from roi_color_dict)
            color: str | tuple[float, float, float]
            if roi_color_dict is not None:
                color = roi_color_dict[roi]
            else:
                color = metric_color_dict[metric]
            y = [alignment_pattern_1[metric][roi][_roi] for _roi in valid_rois]
            ax1.plot(
                range(len(valid_rois)),
                y,
                color=color,  # type: ignore[arg-type]
                linestyle="--",
                label=f"{roi} ap",
                **plot_kwargs,
            )
        predictor_color_dict = {}
        if alignment_pattern_2_type == "model":
            n_predictors = int(len(alignment_pattern_2[metric][roi].keys()))
            predictors = list(alignment_pattern_2[metric][roi].keys())
            palette = sns.color_palette(aps_cmap_str, n_predictors)
            cmap = ListedColormap(palette)
            sm = ScalarMappable(
                cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=n_predictors)
            )
            sorted_aps_values = np.sort(list(aps[metric][roi].values()))
            sorting_idxs = np.argsort(list(aps[metric][roi].values()))
            sorted_predictors = [predictors[idx] for idx in sorting_idxs]
            for predictor in sorted_predictors:
                ordinal_value = np.where(
                    sorted_aps_values == aps[metric][roi][predictor]
                )[0][0]
                predictor_color_dict[predictor] = \
                    sm.to_rgba(np.array([ordinal_value]))[0]
                y = [
                    alignment_pattern_2[metric][roi][predictor][_roi]
                    for _roi in valid_rois
                ]
                axtwinx.plot(
                    range(len(valid_rois)),
                    y,
                    color=sm.to_rgba(np.array([ordinal_value]))[0],  # type: ignore[arg-type]
                    #  linestyle='--',
                    #  label=f"{short_predictor}, {aps[metric][roi][predictor]:.2f}",
                    **plot_kwargs,
                )
            for _roi in valid_rois:
                if not _roi == roi:
                    for model, ap in alignment_pattern_2[metric][_roi].items():

                        ax2.scatter(
                            ap[_roi],
                            aps[metric][_roi][model],
                            color='grey',
                            alpha=0.5,
                            marker='o',
                            zorder=0,
                            **plot_kwargs,
                        )
                else:
                    for model, ap in alignment_pattern_2[metric][_roi].items():
                        ax2.scatter(
                            ap[_roi],
                            aps[metric][_roi][model],
                            color=predictor_color_dict[model],
                            marker='x',
                            zorder=10,
                            **plot_kwargs,
                        )
    # Configure twin axis
    axtwinx.tick_params(axis="y", labelcolor="black")
    axtwinx.spines["right"].set_visible(True)
    ax1.spines["top"].set_visible(False)
    axtwinx.spines["top"].set_visible(False)
    from matplotlib.ticker import MaxNLocator

    ax1.yaxis.set_major_locator(MaxNLocator(nbins=3))
    axtwinx.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax1.set_xticks(range(len(valid_rois)))
    ax1.set_xticklabels(valid_rois, rotation=90)
    ax1.set_xlabel("ROIs")
    ax1.set_ylabel("alignment score")
    # ax2.set_ylabel("model-brain alignment score")
    # Set slight grey background
    # ax1.set_facecolor('#f5f5f5')  # Light grey background
    # ax2.set_facecolor('#f5f5f5')   # Match twin axis background

    ax1.set_title(f"{roi}, {metric}")
    plt.colorbar(sm, cax=axcm)
    axcm.set_yticks(
        [i + 0.5 for i in range(len(alignment_pattern_2[metric][roi].keys()))]
    )

    axcm.set_yticklabels([predictor[:40] for predictor in sorted_predictors])
    # axcm.yaxis.set_label_position("left")
    # axcm.set_ylabel("aps score", )
    axcm.set_title("aps score")
    n_predictors = len(predictors)
    for predictor in predictors:
        ordinal_value = np.where(sorted_aps_values == aps[metric][roi][predictor])[0][0]
        # Find center of each color segment in axis coordinates
        y = (ordinal_value + 0.5) / (n_predictors)
        axcm.text(
            0.5,
            y,
            f"{aps[metric][roi][predictor]:.2f}",
            ha="center",
            va="center",
            color="black",
            fontsize=10,
            transform=axcm.transAxes,
        )
    ax1.legend()

    ax2.set_xlabel(f"{metric} alignment score")
    ax2.set_ylabel(f"{metric} aps score")
    ax2.set_title("rsa score vs. aps score")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["left"].set_visible(True)
    ax2.spines["bottom"].set_visible(True)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    return fig, ax1, ax2


# def plot_alignment_pattern_scores_by_model(
#     alignment_pattern_brain: dict[str, dict[str, dict[str, float]]],
#     alignment_pattern_model: dict[str, dict[str, dict[str, dict[str, float]]]],
#     metric: str,
#     model_key: str,
#     aps: dict[str, dict[str, dict[str, float]]],
#     valid_rois: list[str],
#     roi_color_dict: dict[str, tuple[float, float, float]] | None = None,
#     highlight_alpha: float = 0.15,
#     require_best_predictor: bool = True,
#     **plot_kwargs,
# ) -> tuple[plt.Figure, plt.Axes]:
#     """Plot brain vs model alignment patterns for ROIs best predicted by a model/layer

#     Args:
#         alignment_pattern_brain: Mapping metric -> ROI -> ROI -> alignment score.
#         alignment_pattern_model: Mapping metric -> ROI -> predictor -> ROI -> score.
#         metric: Metric name to visualise (e.g. ``"rsa"``).
#         model_key: Combined model/layer identifier matching keys in ``aps``.
#         aps: Mapping metric -> ROI -> predictor -> APS score.
#         valid_rois: Ordered list of ROIs for the x-axis.
#         roi_color_dict: Optional mapping ROI -> RGBA tuple to colour the curves.
#         highlight_alpha: Alpha used when shading the ROI column for emphasis.
#         require_best_predictor: If True, only ROIs where ``model_key`` is the top
#             predictor (highest APS) are plotted. If False, all ROIs containing
#             ``model_key`` are included.
#         **plot_kwargs: Additional keyword arguments forwarded to ``ax.plot``.

#     Returns:
#         Tuple of (figure, axis).
#     """
#     metric_alignment_brain = alignment_pattern_brain.get(metric, {})
#     metric_alignment_model = alignment_pattern_model.get(metric, {})
#     metric_aps = aps.get(metric, {})

#     matching_rois: list[tuple[str, float]] = []
#     for roi in valid_rois:
#         predictor_scores = metric_aps.get(roi, {})
#         if not predictor_scores or model_key not in predictor_scores:
#             continue
#         if require_best_predictor:
#             best_predictor = max(predictor_scores.items(),
# key=lambda item: item[1])[0]
#             if best_predictor != model_key:
#                 continue
#         matching_rois.append((roi, predictor_scores[model_key]))

#     fig, ax = plt.subplots(figsize=(8, 3.5))

#     default_line_kwargs = {"linewidth": 2.0}
#     default_line_kwargs.update(plot_kwargs)
#     line_kwargs = {k: v for k, v in default_line_kwargs.items() if k != "color"}

#     ax_model = ax.twinx()
#     ax_model.spines["right"].set_visible(True)

#     best_roi_for_model: str | None = None
#     best_roi_model_alignment: dict[str, float] | None = None
#     best_roi_aps_value: float | None = None

#     for roi, aps_value in matching_rois:
#         roi_alignment_brain = metric_alignment_brain.get(roi)
#         if roi_alignment_brain is None:
#             continue

#         color_override = (
#             roi_color_dict.get(roi) if roi_color_dict is not None else None
#         )
#         x_vals = range(len(valid_rois))
#         brain_vals = [roi_alignment_brain.get(_roi, np.nan) for _roi in valid_rois]

#         (brain_line,) = ax.plot(
#             x_vals,
#             brain_vals,
#             color=color_override,
#             label=f"{roi} brain (APS={aps_value:.2f})",
#             alpha=0.5,
#             **line_kwargs,
#         )

#         roi_idx = valid_rois.index(roi)
#         highlight_color = list(mcolors.to_rgba(brain_line.get_color()))
#         highlight_color[3] = highlight_alpha
#         ax.axvspan(
#             roi_idx - 0.5,
#             roi_idx + 0.5,
#             color=highlight_color,
#             zorder=0,
#         )

#         roi_alignment_model = metric_alignment_model.get(roi, {}).get(model_key)
#         if roi_alignment_model is None:
#             continue

#         if (
#             best_roi_model_alignment is None
#             or aps_value > (best_roi_aps_value if best_roi_aps_value\
#                 is not None else -np.inf)
#         ):
#             best_roi_for_model = roi
#             best_roi_model_alignment = roi_alignment_model
#             best_roi_aps_value = aps_value

#     model_line_handle = None
#     if best_roi_model_alignment is not None:
#         model_vals = [
#             best_roi_model_alignment.get(_roi, np.nan) for _roi in valid_rois
#         ]
#         model_color = (
#             roi_color_dict.get(best_roi_for_model, (0.1, 0.1, 0.1, 1.0))
#             if roi_color_dict is not None and best_roi_for_model is not None
#             else (0.1, 0.1, 0.1, 1.0)
#         )
#         (model_line_handle,) = ax_model.plot(
#             range(len(valid_rois)),
#             model_vals,
#             color=model_color,
#             linestyle="--",
#             linewidth=line_kwargs.get("linewidth", 2.0) + 0.5,
#             label=(
#                 f"{best_roi_for_model} model AP (APS={best_roi_aps_value:.2f})"
#                 if best_roi_for_model is not None and best_roi_aps_value is not None
#                 else f"{model_key} model AP"
#             ),
#         )

#     ax.set_title(
#         f"{metric}: {len(matching_rois)} ROI(s) where {model_key} is "
#         f"{'top' if require_best_predictor else 'present'} predictor"
#     )
#     ax.set_xticks(range(len(valid_rois)))
#     ax.set_xticklabels(valid_rois, rotation=90)
#     ax.set_xlabel("ROIs")
#     ax.set_ylabel("brain alignment score")
#     ax_model.set_ylabel("model alignment score")
#     ax.grid(False)

#     legend_handles, legend_labels = ax.get_legend_handles_labels()
#     if model_line_handle is not None:
#         model_handle, model_label = ax_model.get_legend_handles_labels()
#         legend_handles += model_handle
#         legend_labels += model_label

#     if legend_handles:
#         ax.legend(
#             legend_handles,
#             legend_labels,
#             loc="upper left",
#             bbox_to_anchor=(1.2, 1.0),
#             borderaxespad=0.0,
#             frameon=False,
#         )
#     sns.despine()
#     fig.tight_layout()
#     return fig, ax

