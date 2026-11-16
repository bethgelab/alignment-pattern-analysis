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


def align_order_to_available(
    order: List[str],
    available: set,
    source: str = "the reference ordering",
) -> List[str]:
    """Restrict a model ordering to the models actually present.

    Figures that compare two datasets must put the same model at the same x
    position, so the ordering is ranked once on a reference dataset and reused.
    The second dataset may not hold every one of those models - a sweep can
    lose jobs - and ``DataFrame.loc`` would then silently insert all-NaN rows
    rather than raise. Dropping the missing names keeps the relative order,
    which is the whole point of reusing it.

    Parameters
    ----------
    order : list of str
        Model names in the order the bars should appear.
    available : set
        Model names present in the data being plotted.
    source : str
        Where ``order`` came from, for the log message.

    Returns:
    -------
    list of str
        ``order`` with the absent names removed, order otherwise unchanged.

    Raises:
    ------
    ValueError
        If no model in ``order`` is present in the data, which means the two
        aggregations do not describe the same models at all.
    """
    import logging

    logger = logging.getLogger(__name__)

    kept = [model for model in order if model in available]
    missing = [model for model in order if model not in available]
    if missing:
        logger.warning(
            "%d of %d models in %s are absent from these results and are "
            "dropped from the x-axis: %s",
            len(missing),
            len(order),
            source,
            ", ".join(sorted(missing)),
        )
    extra = sorted(available - set(order))
    if extra:
        logger.warning(
            "%d models in these results are absent from %s and cannot be "
            "placed on the shared x-axis: %s",
            len(extra),
            source,
            ", ".join(extra),
        )
    if not kept:
        raise ValueError(
            f"None of the {len(order)} models in {source} are present in "
            "these results, so the two cannot share an x-axis. Check that "
            "both aggregations were run over the same set of models."
        )
    return kept
