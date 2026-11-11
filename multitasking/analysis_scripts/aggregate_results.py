#!/usr/bin/env python3
"""Script to aggregate and analyze bold moments results.

Converted from aggregate_results.ipynb.
"""
import logging
import os
import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from multitasking.analyses.alignment_patterns.base import SIMILARITY_METRICS
from multitasking.analyses.alignment_patterns.brain_brain_alignment_patterns import (
    PairwiseBrainBrainAlignmentPatterns,
)
from multitasking.analyses.alignment_patterns.model_brain_alignment_patterns import (
    ModelBrainAlignmentPatterns,
)
from multitasking.analyses.dataframe_utils import (
    get_order,
    load_bm_results,
    select_top_layer,
)
from multitasking.analyses.equivalence_analysis import run_equivalence_analysis
from multitasking.CONSTANTS import get_rois, get_subjects, parse_roi_names
from multitasking.utils.scoresheet_loading import load_scoresheets_from_parent_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# The repo root, two levels up from multitasking/analysis_scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ATTRIBUTES_PATH = PROJECT_ROOT / "model_attributes.ods"



def bootstrap_rows_by_subject_fast(df, random_state=None):
    """Bootstrap rows by resampling subjects with replacement."""
    rng = np.random.default_rng(random_state)

    subjects = df["subject"].unique()
    n = len(subjects)

    sampled_subjects = rng.choice(subjects, size=n, replace=True)

    # Make a small table of the sampled subjects (with duplicates)
    idx = pd.DataFrame({"subject": sampled_subjects})

    # Merge expands rows automatically to replicate subjects the right number of times
    boot_df = idx.merge(df, on="subject", how="left")

    return boot_df, sampled_subjects



#: Column naming the stimulus set a score was computed on. Written by
#: benchmark.py and intersubject_consistency.py. Scoresheets produced before it
#: existed do not carry it, so every use below degrades gracefully.
STIMULUS_SET_COLUMN = "stimulus_set"


def deduplication_keys(df: pd.DataFrame) -> list[str]:
    """Returns the columns that identify one score.

    ``split`` alone does not: a sweep that scored a model on NSD-synthetic and
    one that scored it on NSD-core's held-out stimuli produce rows that are
    identical under everything else, so de-duplicating without the stimulus set
    would silently drop one of the two.

    Args:
        df: The scoresheet frame.

    Returns:
        The subset to pass to ``drop_duplicates``.
    """
    keys = ["model", "layer", "roi", "metric", "split", "subject"]
    if STIMULUS_SET_COLUMN in df.columns:
        keys.append(STIMULUS_SET_COLUMN)
    return keys


def check_stimulus_sets_align(
    scores_df: pd.DataFrame, ceiling_df: pd.DataFrame, what: str
) -> None:
    """Checks that scores and noise ceilings describe the same stimuli.

    The noise-ceiling merge joins on ``roi``, ``split``, ``subject`` and
    ``metric``, none of which say which stimuli were shown. A model scored on
    NSD-synthetic would merge happily against an NSD-core ceiling and be
    normalised by it, with nothing raising.

    Args:
        scores_df: The model or pairwise scores.
        ceiling_df: The noise ceilings being merged in.
        what: Name of the ceiling, for the error message.

    Raises:
        ValueError: If the two disagree on the stimulus set within a split.
    """
    if (
        STIMULUS_SET_COLUMN not in scores_df.columns
        or STIMULUS_SET_COLUMN not in ceiling_df.columns
    ):
        logger.warning(
            "Scoresheets predate the %r column, so the %s cannot be checked "
            "against the stimuli the scores were computed on. Re-run the sweep "
            "to get that guarantee.",
            STIMULUS_SET_COLUMN,
            what,
        )
        return

    for split in sorted(set(scores_df["split"]) & set(ceiling_df["split"])):
        in_scores = set(
            scores_df.loc[scores_df["split"] == split, STIMULUS_SET_COLUMN]
            .dropna()
            .unique()
        )
        in_ceiling = set(
            ceiling_df.loc[ceiling_df["split"] == split, STIMULUS_SET_COLUMN]
            .dropna()
            .unique()
        )
        if in_scores and in_ceiling and in_scores != in_ceiling:
            raise ValueError(
                f"In the {split!r} split, the scores were computed on "
                f"{sorted(in_scores)} but the {what} on {sorted(in_ceiling)}. "
                "Normalising one by the other would be meaningless. Re-run the "
                "inter-subject sweep with the same config as the benchmark "
                "sweep, or aggregate the two stimulus sets separately."
            )


@click.command()
@click.option(
    '--output-supdir',
    help='Path to BM results directory',
    default="/",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    '--intersubject-dir',
    default="/",
    help='Path to intersubject results directory',
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    '--pairwise-subject-dir',
    default="/",
    help='Path to pairwise subject results directory',
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    '--aps-similarity-metric',
    default='pearson',
    show_default=True,
    help=(
        "Similarity metric for comparing alignment patterns. 'mse' is the "
        "mean squared error between two patterns, reported negated so that "
        "larger stays 'more similar' for every metric: the best model is "
        "still the maximum and the equivalence bands still open downwards. "
        "Unlike 'pearson' it is sensitive to scale and offset, so its values "
        "are only comparable within one --main-outcome. The pickles this "
        "script writes are NOT named after this choice, so run a second "
        "metric into a different $SCRATCH or it overwrites the first."
    ),
    type=click.Choice(list(SIMILARITY_METRICS)),
)
@click.option(
    '--main-outcome',
    default='score',
    help='Main outcome to use for selecting top layer (default: score)',
    type=click.Choice(['score', 'score_norm_upper', 'score_norm_lower']),
)
@click.option(
    "--dataset",
    type=click.Choice(["bold_moments", "nsd", "nsd_synthetic"]),
    default="bold_moments",
    show_default=True,
    help="Which dataset's canonical ROI and subject lists to use.",
)
@click.option(
    "--model-attributes-path",
    default=str(DEFAULT_MODEL_ATTRIBUTES_PATH),
    show_default=True,
    help=("Spreadsheet of per-model attributes (architecture, training data, ...) "
          "merged into the output dataframe. Reading it needs odfpy."),
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--exclude-rois",
    type=str,
    default="",
    help=(
        "Comma-separated ROI names to drop entirely, e.g. 'V6,V6A'. An "
        "alignment pattern is a vector over the retained ROIs, so excluding "
        "one changes every pattern, not only that ROI's entry."
    ),
)
def main(dataset: str,
         output_supdir: Path,
         intersubject_dir: Path,
         pairwise_subject_dir: Path,
         aps_similarity_metric: str,
         model_attributes_path: Path,
         exclude_rois: str,
         main_outcome: str = "score"):
    """Main function to aggregate and analyze results."""
    excluded = parse_roi_names(exclude_rois)
    rois = get_rois(dataset, exclude=excluded)
    subjects = get_subjects(dataset)
    if excluded:
        logger.info(f"Excluding ROIs {excluded}; {len(rois)} remain: {rois}")
    SCRATCH = Path(os.environ.get("SCRATCH", "/scratch"))

    # Load and organize results
    full_df = load_bm_results(output_supdir)

    # filter out rois not in the constant ROIS
    full_df = full_df[full_df["roi"].isin(rois)]

    full_df.drop_duplicates(subset=deduplication_keys(full_df), inplace=True)

    full_df["predictor"] = pd.concat([full_df["model"], full_df["layer"]], axis=1
                                       ).apply(lambda x: "__".join(x), axis=1)
    subjects = sorted(full_df["subject"].unique())
    metrics = sorted(full_df["metric"].unique())

    # Load intersubject data
    intersubject_df = load_scoresheets_from_parent_dir(intersubject_dir)

    # Load pairwise subject data
    pairwise_subject_df = load_scoresheets_from_parent_dir(pairwise_subject_dir)

    logger.info("Checking for duplicates in pairwise subject dataframe")
    # Strip pairwise of duplicates and unused ROIs
    assert len(pairwise_subject_df[pairwise_subject_df.duplicated(keep=False)\
               & ~(pairwise_subject_df["roi"] == "V7") \
                   & ~(pairwise_subject_df["layer"] == "V7")]) == 0, \
                       "Pairwise subject dataframe has duplicates other than V7/V7" # V7 was duplicate in config
    logger.info("No unexpected duplicates found in pairwise subject dataframe")
    pairwise_subject_df.drop_duplicates(
        subset=deduplication_keys(pairwise_subject_df), inplace=True
    )

    # filter out rois not in the constant ROIS
    pairwise_subject_df = pairwise_subject_df.query("roi in @rois and layer in @rois")

    # Calculate upper noise ceiling for each ROI
    tmp_df = intersubject_df.query("layer == roi")
    ceiling_columns = ["roi", "split", "subject", "metric", "score"]
    if STIMULUS_SET_COLUMN in tmp_df.columns:
        ceiling_columns.append(STIMULUS_SET_COLUMN)
    noise_ceiling_df = (
        tmp_df
        .loc[:, ceiling_columns]
        .rename(columns={"score": "upper_noise_ceiling"})
    )
    assert not noise_ceiling_df.duplicated(
        ["roi", "split", "subject", "metric"]
    ).any()

    check_stimulus_sets_align(
        pairwise_subject_df, intersubject_df, "upper noise ceiling"
    )
    pairwise_subject_df = pairwise_subject_df.merge(
        noise_ceiling_df.drop(columns=[STIMULUS_SET_COLUMN], errors="ignore"),
        on=["roi", "split", "subject", "metric"],
        how="left"
    )

    # Calculate lower noise ceiling for each ROI
    tmp_df = pairwise_subject_df.query("layer == roi")
    lower_noise_ceiling_df = tmp_df.groupby(
        ["roi", "split", "metric", "subject"], as_index=False
        ).agg(
            lower_noise_ceiling = ("score", "mean"),
            lower_noise_ceiling_std = ("score", "std"),
            lower_noise_ceiling_count = ("score", "count"),
            lower_noise_ceiling_sem = ("score", "sem"),
            )
    assert not lower_noise_ceiling_df.duplicated(
        ["roi", "split", "subject", "metric"]
    ).any()

    pairwise_subject_df = pairwise_subject_df.merge(
        lower_noise_ceiling_df,
        on=["roi", "split", "subject", "metric"],
        how="left"
    )


    # Normalise scores by noise ceilings
    pairwise_subject_df["score_norm_upper"] = \
        pairwise_subject_df["score"] / pairwise_subject_df["upper_noise_ceiling"]
    pairwise_subject_df["score_norm_lower"] = \
        pairwise_subject_df["score"] / pairwise_subject_df["lower_noise_ceiling"]

    check_stimulus_sets_align(full_df, intersubject_df, "upper noise ceiling")
    check_stimulus_sets_align(
        full_df, pairwise_subject_df, "lower noise ceiling"
    )

    full_df = full_df.merge(
        noise_ceiling_df.drop(columns=[STIMULUS_SET_COLUMN], errors="ignore"),
        on=["roi", "split", "subject", "metric"],
        how="left",
        validate="m:1", # many to one merge
    )
    full_df = full_df.merge(
        lower_noise_ceiling_df,
        on=["roi", "split", "subject", "metric"],
        how="left",
        validate="m:1", # many to one merge
    )
    full_df = full_df.reset_index(drop=True)

    # Normalise scores by noise ceilings
    full_df["score_norm_upper"] = (
        np.asarray(full_df["score"]) / np.asarray(full_df["upper_noise_ceiling"])
    )
    full_df["score_norm_lower"] = (
        np.asarray(full_df["score"]) / np.asarray(full_df["lower_noise_ceiling"])
    )

    pickle.dump(full_df, open(SCRATCH / "full_df_extended.pkl", "wb"))

    ############################################################
    # APS analysis
    ############################################################

    # Do alignment pattern similarity analysis
    logger.info(f"Computing brain-brain alignment patterns using similarity metric {aps_similarity_metric}...")
    model_brain_similarities = []
    for metric in metrics:
        pw_brain_brain_alignment_patterns = PairwiseBrainBrainAlignmentPatterns(
            metric=metric,
            main_outcome=main_outcome
        )
        pw_brain_brain_alignment_patterns.get_alignment_pattern_df(pairwise_subject_df,
                                                                   split="test")
        pw_brain_brain_alignment_patterns.get_brain_brain_alignment_pattern_similarity(
            rois=rois,
            subjects=subjects,
            similarity_metric=aps_similarity_metric
        )
        logger.info(f"Brain-brain alignment patterns similarity computed for "
                    f"{len(pw_brain_brain_alignment_patterns.alignment_pattern_similarity_df)} rows")
        #########################################################

        # Model-brain alignment patterns
        logger.info(f"Computing model-brain alignment patterns using similarity metric {aps_similarity_metric}...")
        model_brain_alignment_patterns = ModelBrainAlignmentPatterns(
            metric=metric,
            main_outcome=main_outcome
        )

        model_brain_alignment_patterns.get_alignment_pattern_df(full_df, split="test")
        model_brain_similarities_df = \
            model_brain_alignment_patterns.get_alignment_pattern_similarity(
            subjects=subjects,
            predictors=sorted(full_df["predictor"].unique()),
            rois=rois,
            brain_brain_alignment_patterns=pw_brain_brain_alignment_patterns,
            similarity_metric=aps_similarity_metric
        )

        logger.info(f"Model-brain alignment patterns similarity computed for "
                    f"{len(model_brain_alignment_patterns.alignment_pattern_similarity_df)} rows")

        model_brain_similarities.append(model_brain_similarities_df)
        pickle.dump(
            model_brain_alignment_patterns,
            open(SCRATCH / f"model_brain_alignment_patterns_{metric}_{main_outcome}.pkl", "wb")
            )
        pickle.dump(
            pw_brain_brain_alignment_patterns,
            open(SCRATCH / f"pw_brain_brain_alignment_patterns_{metric}_{main_outcome}.pkl", "wb")
            )

    full_df = full_df.merge(pd.concat(model_brain_similarities, ignore_index=True),
                            on=["roi", "subject", "predictor", "metric", "split"],
                            how="left")

    # Subject averaging

    subject_avg_df = full_df.groupby(
        ["model", "layer", "roi", "split", "metric"], as_index=False
    ).agg(
        count=("score", "count"),
        score_mean=("score", "mean"),
        score_std=("score", "std"),
        score_norm_upper_mean=("score_norm_upper", "mean"),
        score_norm_upper_std=("score_norm_upper", "std"),
        score_norm_lower_mean=("score_norm_lower", "mean"),
        score_norm_lower_std=("score_norm_lower", "std"),
        score_norm_upper_count=("score_norm_upper", "count"),
        score_norm_lower_count=("score_norm_lower", "count"),
        score_norm_upper_sem=("score_norm_upper", "sem"),
        score_norm_lower_sem=("score_norm_lower", "sem"),
        aps_score_mean=("similarity", "mean"),
        aps_score_std=("similarity", "std"),
    )

    # Select best layer for each model and ROI based on train split and avg. main outcome
    df, roi_layer_map = select_top_layer(subject_avg_df, split="train",
                                         column_name=f"{main_outcome}_mean")
    roi_layer_map["is_best_layer"] = True

    # Add best layer information to full dataframe
    full_df = full_df.merge(roi_layer_map,
                            on=["roi", "layer", "metric", "model"],
                            how="left",
                            validate="m:1"
        )
    full_df["is_best_layer"] = full_df["is_best_layer"].fillna(False)
    # Add best layer information to subject averaged dataframe
    subject_avg_df = subject_avg_df.merge(roi_layer_map,
                         on=["roi", "layer", "metric", "model"],
                         how="left",
                         validate="m:1")
    subject_avg_df["is_best_layer"] = subject_avg_df["is_best_layer"].fillna(False)

    # Get model ordering as the average score across all ROIs
    model_orders = {}
    for metric in metrics:
        model_orders[f"{metric}"] = get_order(
                    df,
                    metric=metric,
                    order_by="model",
                    column_name=f"{main_outcome}_mean"
                    )
        df[f"model_order_{metric}"] = pd.Categorical(
            df["model"],
            categories=model_orders[f"{metric}"],
            ordered=True)
        full_df[f"model_order_{metric}"] = pd.Categorical(
            full_df["model"],
            categories=model_orders[f"{metric}"],
            ordered=True)

    # loop once over all combinations
    for metric in metrics:
        for roi in full_df["roi"].unique():

            # ---- main outcome best model ----
            order_main = get_order(
                df, # is a dataframe with the top layer for each model and ROI
                metric=metric,
                additional_condition=(df["roi"] == roi),
                order_by="model",
                column_name=f"{main_outcome}_mean",
            )
            best_model_main = order_main[-1]

            # ---- APS best model ----
            order_aps = get_order(
                df,
                metric=metric,
                additional_condition=(df["roi"] == roi),
                order_by="model",
                column_name="aps_score_mean",
            )
            best_model_aps = order_aps[-1]


            # ---- mark both in both dataframes ----
            for target_df in (full_df, subject_avg_df):

                base_mask = (
                    (target_df["metric"] == metric)
                    & (target_df["roi"] == roi)
                    & (target_df["split"] == "test")
                    & (target_df["is_best_layer"])
                )

                target_df.loc[
                    base_mask & (target_df["model"] == best_model_main),
                    "is_best_model",
                ] = True

                target_df.loc[
                    base_mask & (target_df["model"] == best_model_aps),
                    "is_best_model_aps",
                ] = True


    # finalize boolean columns
    full_df["is_best_model"] = full_df["is_best_model"].fillna(False)
    subject_avg_df["is_best_model"] = subject_avg_df["is_best_model"].fillna(False)

    full_df["is_best_model_aps"] = full_df["is_best_model_aps"].fillna(False)
    subject_avg_df["is_best_model_aps"] = subject_avg_df["is_best_model_aps"].fillna(False)

    # Save intermediate results


    pickle.dump(intersubject_df, open(SCRATCH / "intersubject_df.pkl", "wb"))
    pickle.dump(pairwise_subject_df, open(SCRATCH / "pairwise_subject_df.pkl", "wb"))
    pickle.dump(full_df, open(SCRATCH / "full_df.pkl", "wb"))

    # Bootstrap analysis

    n_bootstrap_samples = 1000
    sampled_subjects_all = []

    bootstrap_restriction = "split == 'test' and is_best_layer"

    bootstrap_rows = []

    for bs_idx in tqdm(range(n_bootstrap_samples)):

        boot_df, sampled_subjects = bootstrap_rows_by_subject_fast(
            full_df.query(bootstrap_restriction),
            random_state=bs_idx
        )
        sampled_subjects_all.append(sampled_subjects)

        subject_avg_df_bs = (
            boot_df
            .groupby(["model", "layer", "roi", "split", "metric"], as_index=False)
            .agg(
                score_mean=(main_outcome, "mean"),
                aps_score_mean=("similarity", "mean"),
            )
        )

        subject_avg_df_bs["bootstrap"] = bs_idx

        bootstrap_rows.append(
            subject_avg_df_bs[
                ["bootstrap", "metric", "roi", "model", "score_mean", "aps_score_mean"]
            ]
        )


    # combine all bootstrap samples
    bootstrap_all = pd.concat(bootstrap_rows, ignore_index=True)

    ci_main = (
        bootstrap_all
        .groupby(["metric", "roi", "model"])["score_mean"]
        .quantile([0.025, 0.975])
        .unstack()
    )

    ci_main[f"{main_outcome}_ci"] = list(zip(ci_main[0.025], ci_main[0.975], strict=False))
    ci_main = ci_main[[f"{main_outcome}_ci"]]


    ci_aps = (
        bootstrap_all
        .groupby(["metric", "roi", "model"])["aps_score_mean"]
        .quantile([0.025, 0.975])
        .unstack()
    )

    ci_aps["aps_score_ci"] = list(zip(ci_aps[0.025], ci_aps[0.975], strict=False))
    ci_aps = ci_aps[["aps_score_ci"]]


    ci_df = ci_main.join(ci_aps).reset_index()
    ci_df["is_best_layer"] = True

    subject_avg_df = subject_avg_df.merge(ci_df, on=["roi", "model", "metric",
                                                     "is_best_layer"], how="left")
    logger.info(f"Subject avg df after merging CI: {subject_avg_df.shape}")
    pickle.dump(subject_avg_df, open(SCRATCH / f"subject_avg_df_{main_outcome}.pkl", "wb"))
    subject_avg_df = run_equivalence_analysis(subject_avg_df, main_outcome=main_outcome)


    ############################################################
    # Add model attributes
    ############################################################
    # Load model attributes
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
    df.insert(
        df.shape[1],
        "model_provider",
        df["model"].apply(lambda x: x.split('/')[0]))

    ############################################################


    pickle.dump(full_df, open(SCRATCH / "full_df_extended.pkl", "wb"))
    pickle.dump(df, open(SCRATCH / f"df_{main_outcome}.pkl", "wb"))
    pickle.dump(subject_avg_df, open(SCRATCH / f"subject_avg_df_equiv_{main_outcome}.pkl", "wb"))


    with open(SCRATCH / f"bootstrap_all_{main_outcome}.pkl", "wb") as f:
        pickle.dump(bootstrap_all, f)

    logger.info("Analysis complete! Results saved to SCRATCH directory.")


if __name__ == "__main__":
    main()

