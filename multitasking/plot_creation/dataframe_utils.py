from typing import List, Union

import pandas as pd
from pathlib import Path


def load_bm_results(output_supdir: Path) -> pd.DataFrame:
    """Load the results from the boldmoments analysis.

    Parameters
    ----------
    output_supdir : Path
        Path to the results directory.

    Returns
    -------
    pd.DataFrame
        DataFrame containing all loaded scoresheets. Returns empty DataFrame
        if no scoresheets are found.
    """
    all_scoresheets = []
    # Recursively find all scoresheet CSV files in all subfolders
    scorefiles = list(Path(output_supdir).rglob("scoresheets/*.csv"))
    for fpath in scorefiles:
        scoresheet = pd.read_csv(fpath)
        all_scoresheets.append(scoresheet)
    
    if not all_scoresheets:
        return pd.DataFrame()
    
    full_df = pd.concat(all_scoresheets, ignore_index=True)
    return full_df
    


def add_stream_hierarchy_info(df: pd.DataFrame, add_new_rois: bool = False,) -> pd.DataFrame:
    """Add stream and hierarchy information to a DataFrame based on ROI names.

    This function classifies ROIs into visual processing streams (early, dorsal,
    ventral)
    and adds hierarchy information based on known anatomical connections between
    visual areas (from Rolls 2022)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'roi' column with region names to classify.

    Returns:
    -------
    pd.DataFrame
        Original DataFrame with additional columns:
        - 'stream': Classification of ROI into 'early', 'dorsal', 'ventral', or
        'unknown'
        - 'order': Hierarchy assignment
    """
    early = {"V1", "V2", "V3"}
    dorsal_rois = {
        "PH",
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
        # "RSC",
        "IPS1",
    }
    ventral_rois = {"V4", "V8", "FFC", "PIT"}

    if add_new_rois:
        raise NotImplementedError("New ROIs not implemented yet")
        # early = {"V1", "V2", "V3", "ProS"}
        # ventral_rois = {"V4", "V8", "PIT", "FFC", "VVC", "VMV1", "VMV2", "VMV3", "PHA2", "PHA3"}
        # dorsal_rois = {"PH", "V6", "V7", "IPS1", "MT", "MST", "FST", "LO1", "LO2", "LO3", "V3A", "V3B", "V6A", "LIPd", "LIPv", "AIP", "7PC"}



    def classify_stream(region: str) -> str:
        if region in dorsal_rois:
            return "dorsal"
        elif region in ventral_rois:
            return "ventral"
        elif region in early:
            return "early"
        else:
            return "unknown"

    connectivity_dict: dict[str, tuple[tuple[str, ...], tuple[str, ...], int]] = {
        "V1": ((), ("V2",), 1),
        "V2": (("V1",), ("V3",), 2),
        "V3": (("V2",), ("V4", "LO3", "V3A", "V3B", "MT"), 3),
        "V4": (("V3",), ("V8", "PIT"), 4),
        "V8": (("V4",), ("FFC",), 5),
        "PIT": (("V4",), (), 5),
        "FFC": (("V8",), (), 6),  # higher-level ventral stream
        "V3A": (("V3",), ("V6A",), 4),
        "V3B": (("V3",), ("V6A",), 4),
        "V6A": (("V3A", "V3B", ), (), 5),
        "MT": (("V3", "LO3"), ("MST",), 4),
        "MST": (("MT",), ("FST",), 5),
        "FST": (("MST",), (), 6),  # dorsal extrastriate
        "LO3": (("V3",), ("MT",), 4),
    }



    if add_new_rois:
        connectivity_dict = {
            "V1": ((), ("V2",), 1),

            "V2": (("V1",), ("V3", "ProS",), 2),

            "V3": (("V2",), ("V4", "LO3", "V3A", "V3B", "MT",), 3),

            "V4": (("V3",), ("V8", "PIT", "FFC", "VMV3",), 4),    # from Fig 1 (caption) in Rolls

            "V8": (("V4",), ("FFC", "VMV3",), 5),                # check Fig 3 in Rolls

            "PIT": (("V4",), ("PH", "FFC",), 5),          # Fig 1 in Rolls

            "FFC": (("V4", "V8", "PIT",), ("TE2p", "PH", "VVC",), 6),  # Fig 1.

            "PH": (("FFC", "PIT", "FST"), ("AIP"), 6),               # ? hierarchy uncertain

            "V3A": (("V3",), ("V6A", "V7",), 4),          # Rolls Fig. 6

            "V3B": (("V3",), ("V6A", "V7",), 4),          # Rolls Fig. 6

            # "V6": (("V2", "V3"), ("V6A",), 4),         # insufficient data

            "V7": (("V3A", "V3B",), ("V6A", "IPS1", "LIPv",), 4),

            "V6A": (("V3A", "V3B", "V7",), ("IPS1",), 5), # Rolls Fig. 6

            "MT": (("V3", "LO3",), ("MST", "LO1", "LO2", "LO3", "FFC",), 4),  # Rolls Fig. 6, Fig. 4

            "MST": (("MT",), ("FST", "VMV1"), 5),               # Rolls Fig. 4

            "LO3": (("V3",), ("MT",), 5),

            "FST": (("MST", "LO2",), ("LO1", "LO2", "LIPd", "LIPv", "AIP",), 6),  # Rolls Fig. 4, Fig. 6

            # "LO1": (("MT", "MST", "FST", "LO2", "LO3"),
            #          ("MT", "MST", "FST", "LO2", "LO3"), 5),
            #          # we just assume connectivity between these regions

            "LO2": (("MT", "MST", "FST", "LO1", "LO3",),
                    ("MT", "MST", "FST", "LO1", "LO3",), 5),  # assumed connectivity

            # "IPS1": (("V3A", "V3B", "V6A", "V7", "LO3"), (), 6),  # insufficient data

            #"IP0": ?,

            # "IP1" : ?,

            #   "MIP" : ?,

            #   "VIP" : ?,

            "LIPd" : (("FST",) , (), 7),  # from Fig 4

            "LIPv" : (("V7", ), ("7PC",), 5,), # from Fig 4

            "AIP": (("FST", "PH"), (), 7), # from Fig 4

            # "7Pm" : ?,

            # "7PL" : ?,

            "7PC" : (("LIPv", "FST",), (), 6), # from Fig 4

            # "7Am", ?

            # "7AL", ?

            "VVC": (("ProS", "FFC",), ("PHA2", "PHA3", "VMV3",), 4),

            "VMV2": (("ProS",), ("PHA2", "PHA3",), 4),

            "VMV3": (("ProS", "V4", "V8", "VVC"), ("PHA2", "PHA3",), 4),

            "VMV1": (("ProS", "MST"), ("PHA2", "PHA3",), 4),

            "PHA2": (("VVC", "VMV1", "VMV2", "VMV3",), (), 5),

            "PHA3": (("VVC", "VMV1", "VMV2", "VMV3",), (), 5),

            "ProS" : (("V2",), ("VVC", "VMV1", "VMV2", "VMV3",), 3), # from Fig 6 in Rolls

            #  "PH"
        }



    def assign_order(region: str) -> int:
        order = connectivity_dict.get(region, ((), (), -1))
        return order[2]

    df.insert(df.shape[1], "stream", df["roi"].apply(classify_stream))
    # throw out non-visual rois
    df = df[df["stream"].isin(["early", "ventral", "dorsal", "new"])]

    df.insert(df.shape[1], "order", df["roi"].apply(assign_order))
    return df


def select_top_layer(df: pd.DataFrame, split: str,
                     column_name: str = "score_mean") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the top performing layer for each model-ROI-metric combination.

    This function identifies the layer with the highest score for each combination
    of model, ROI, and metric within a specified split, then filters the dataframe
    to only include those top-performing layers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing model performance data with columns:
        - 'split': data split (e.g., 'train', 'test')
        - 'roi': region of interest
        - 'metric': evaluation metric (e.g., 'rsa', 'linear_predictivity')
        - 'model': model name
        - 'layer': layer identifier
        - 'score_mean': mean performance score
    split : str
        The data split to use for selecting top layers (e.g., 'train', 'test')

    Returns:
    -------
    pd.DataFrame
        Filtered DataFrame containing only the top-performing layer for each
        model-ROI-metric combination, sorted by ROI, split, and metric.
    """
    top_layers_split = (
        df[df["split"] == split]
        .groupby(["roi", "metric", "model"])[column_name]
        .idxmax()
    )
    if any(pd.isna(top_layers_split)):
        N = len(top_layers_split)
        top_layers_split = top_layers_split.dropna()
        import logging
        LOGGER = logging.getLogger(__name__)
        LOGGER.warning(f"Dropped {N - len(top_layers_split)} rows with NA values in top_layers_split")


    roi_layer_map = df.loc[top_layers_split, ["roi", "layer", "metric", "model"]]

    result_df = df.merge(
        roi_layer_map, on=["roi", "layer", "metric", "model"], how="inner"
    )

    df = result_df.sort_values(["roi", "split", "metric"]).reset_index(drop=True)

    return df, roi_layer_map


def get_order(
    df: pd.DataFrame,
    column_name: str = "score_mean",
    order_by: str = "model",
    metric: str = "rsa",
    additional_condition: Union[bool, pd.Series] = True,
) -> List[str]:
    """Get the ordering of models or ROIs based on their average performance scores.

    This function calculates the average performance score for each model or ROI
    within a specified metric and split, then returns them ordered from lowest
    to highest performance. This ordering is typically used for consistent
    visualization across different plots.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing model performance data with columns:
        - 'split': data split (e.g., 'train', 'test')
        - 'metric': evaluation metric (e.g., 'rsa', 'linear_predictivity')
        - 'score_mean': mean performance score
        - model/roi columns depending on order_by parameter
    column_name : str, optional
        The column to use for calculating averages (default: 'score_mean')
    order_by : str, optional
        The column to group by and order (default: 'model'). Can be 'model' or 'roi'
    metric : str, optional
        The metric to filter by (default: 'rsa')
    additional_condition : bool or pd.Series, optional
        Additional boolean condition to filter the dataframe (default: True)

    Returns:
    -------
    list
        List of model names or ROI names ordered from lowest to highest
        average performance score for the specified metric and split.
    """
    grouped_avg_df = (
        df[(df["split"] == "test") & (df["metric"] == metric) & (additional_condition)]
        .groupby(order_by, as_index=False)[column_name] # group by {model, roi}
        .mean() # calculate mean within a group, i.e. avg for model or roi
    )
    grouped_avg_sorted_df = grouped_avg_df.sort_values(by=column_name) # type: ignore
    return grouped_avg_sorted_df[order_by].to_list()



def add_model_attributes(df: pd.DataFrame) -> pd.DataFrame:
    ############################################################
    # Add model attributes
    ############################################################
        # Load model attributes
    model_attributes_path = Path("/mnt/lustre/work/bethge/bkr578/projects/multitasking_201125/model_attributes.ods")
    model_attributes = pd.read_excel(model_attributes_path, engine="odf")

    # Merge the model attributes into main DF
    # Clean attributes table: ignore rows with NaN model names
    model_attr_clean = model_attributes.dropna(subset=["model", "architecture"]).copy()

    # Strip trailing "*" so we can match prefixes like "clip/ViT"
    model_attr_clean["model_prefix"] = model_attr_clean["model"].str.replace("*", "", regex=False)

    # Build a lookup dict from prefix → row of attributes
    attr_lookup = {
        row["model_prefix"]: row.drop(["model", "model_prefix"]).to_dict()
        for _, row in model_attr_clean.iterrows()
    }

    # Helper: find attributes for a given model string
    def get_attributes(model_name):
        for prefix, attrs in attr_lookup.items():
            if isinstance(model_name, str) and model_name.startswith(prefix):
                return attrs
        return {}

    # Expand attributes into new columns
    attr_expanded = df["model"].apply(get_attributes).apply(pd.Series)

    # Concatenate with original df
    df = pd.concat([df, attr_expanded], axis=1)

    # Add dataset size
    dataset_size_dict = {
        "CLIP": 400000000,
        "Kinetics-400": 306245,
        "Taskonomy": 4500000,
        "ImageNet-1K": 1000000,
        "V-JEPA2": 22000000,
        "VGG-T": 50000000
    }

    df['dataset_size'] = df['dataset'].apply(lambda x: dataset_size_dict[x])
    return df