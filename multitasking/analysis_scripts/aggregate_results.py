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

from multitasking.analyses.alignment_patterns.brain_brain_alignment_patterns import (
    PairwiseBrainBrainAlignmentPatterns,
)
from multitasking.analyses.alignment_patterns.model_brain_alignment_patterns import (
    ModelBrainAlignmentPatterns,
)
from multitasking.analyses.equivalence_analysis import run_equivalence_analysis
from multitasking.CONSTANTS import ROIS as rois
from multitasking.plot_creation.dataframe_utils import (
    get_order,
    load_bm_results,
    select_top_layer,
)
from multitasking.utils.scoresheet_loading import load_scoresheets_from_parent_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def classify_stream(region: str, roi_stream: dict) -> str | None:
    """Classify a region into a stream (early, ventral, or dorsal)."""
    if region in roi_stream['early']:
        return 'early'
    elif region in roi_stream['ventral']:
        return 'ventral'
    elif region in roi_stream['dorsal']:
        return 'dorsal'
    return None


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



@click.command()
@click.option(
    '--output-supdir',
    help='Path to BM results directory',
    default="/mnt/lustre/work/bethge/mwe467/taskonomy/multitasking/output/bold_moments/bm",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    '--intersubject-dir',
    default="/mnt/lustre/work/bethge/bkr857/projects/multitasking/output/bold_moments_intersubject_redo_larger_alpha_range/output/",
    help='Path to intersubject results directory',
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    '--pairwise-subject-dir',
    default="/mnt/lustre/work/bethge/bkr857/projects/multitasking/output/bold_moments_intersubject_pairwise_rsa_lp_5x/output/",
    help='Path to pairwise subject results directory',
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    '--aps-similarity-metric',
    default='pearson',
    help='Similarity metric to use for alignment pattern similarity (default: pearson)',
    type=click.Choice(['pearson', 'mse', 'mae', 'rank_correlation', 'cosine', 'variance_explained']),
)
@click.option(
    '--main-outcome',
    default='score',
    help='Main outcome to use for selecting top layer (default: score)',
    type=click.Choice(['score', 'score_norm_upper', 'score_norm_lower']),
)
def main(output_supdir: Path,
         intersubject_dir: Path,
         pairwise_subject_dir: Path,
         aps_similarity_metric: str,
         main_outcome: str = "score"):
    """Main function to aggregate and analyze results."""
    SCRATCH = Path(os.environ.get("SCRATCH", "/scratch"))

    # Load and organize results
    full_df = load_bm_results(output_supdir)

    # filter out rois not in the constant ROIS
    full_df = full_df[full_df["roi"].isin(rois)]

    # Define ROI assignments:
    roi_stream = {
        "early": ["V1", "V2", "V3"],
        "ventral": ["V4", "V8", "PIT", "FFC"],
        "dorsal": ["PH", "V6", "V7", "IPS1", "MT", "MST", "FST", "LO1", "LO2", "LO3", "V3A", "V3B", "V6A"]
    }

    full_df.insert(full_df.shape[1], "stream", full_df["roi"].apply(lambda x: classify_stream(x, roi_stream)))

    full_df.drop_duplicates(subset=["model", "layer", "roi", "metric", "split", "subject"], inplace=True)

    full_df["predictor"] = pd.concat([full_df["model"], full_df["layer"]], axis=1
                                       ).apply(lambda x: "__".join(x), axis=1)
    subjects = sorted(full_df["subject"].unique())

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
    pairwise_subject_df.drop_duplicates(subset=["model", "layer", "roi", "metric", "split", "subject"],
                                        inplace=True)
    logger.info("No unexpected duplicates found in pairwise subject dataframe")
    pairwise_subject_df = pairwise_subject_df.query("roi in @rois and layer in @rois")


    # Calculate upper noise ceiling for each ROI
    tmp_df = intersubject_df.query("layer == roi")
    noise_ceiling_df = (
        tmp_df
        .loc[:, ["roi", "split", "subject", "metric", "score"]]
        .rename(columns={"score": "upper_noise_ceiling"})
    )
    assert not noise_ceiling_df.duplicated(
        ["roi", "split", "subject", "metric"]
    ).any()

    pairwise_subject_df = pairwise_subject_df.merge(
        noise_ceiling_df,
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

    pickle.dump(full_df, open(SCRATCH / "full_df_no_norm.pkl", "wb"))


    # Normalise scores by noise ceilings
    pairwise_subject_df["score_norm_upper"] = \
        pairwise_subject_df["score"] / pairwise_subject_df["upper_noise_ceiling"]
    pairwise_subject_df["score_norm_lower"] = \
        pairwise_subject_df["score"] / pairwise_subject_df["lower_noise_ceiling"]

    full_df = full_df.merge(
        noise_ceiling_df,
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

    # Normalise scores by noise ceilings (use .values to avoid index alignment issues)
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
    for metric in ["rsa", "linear_predictivity"]:
        pw_brain_brain_alignment_patterns = PairwiseBrainBrainAlignmentPatterns(
            metric=metric,
            main_outcome=main_outcome
        )
        pw_brain_brain_alignment_patterns.get_alignment_pattern_df(pairwise_subject_df, split="test")
        _ = pw_brain_brain_alignment_patterns.get_brain_brain_alignment_pattern_similarity(
            rois=rois,
            subjects=sorted(full_df["subject"].unique()),
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
        logger.info("Computing model-brain alignment patterns similarity...")
        model_brain_similarities_df = model_brain_alignment_patterns.get_alignment_pattern_similarity(
            subjects=subjects,
            predictors=sorted(full_df["predictor"].unique()),
            rois=rois,
            brain_brain_alignment_patterns=pw_brain_brain_alignment_patterns,
            similarity_metric=aps_similarity_metric
        )
        logger.info(f"Model-brain alignment patterns similarity computed for "
                    f"{len(model_brain_alignment_patterns.alignment_pattern_similarity_df)} rows")

        model_brain_similarities.append(model_brain_similarities_df)
        pickle.dump(model_brain_alignment_patterns, open(SCRATCH / f"model_brain_alignment_patterns_{metric}_{main_outcome}.pkl", "wb"))
        pickle.dump(pw_brain_brain_alignment_patterns, open(SCRATCH / f"pw_brain_brain_alignment_patterns_{metric}_{main_outcome}.pkl", "wb"))

    full_df = full_df.merge(pd.concat(model_brain_similarities, ignore_index=True),
                            on=["roi", "subject", "predictor", "metric", "split"],
                            how="left")

    # Subject averaging and layer selection
    subject_avg_df = full_df.groupby(
        ["model", "layer", "roi", "split", "metric", "stream"], as_index=False
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

    df, roi_layer_map = select_top_layer(subject_avg_df, split="train",
                                         column_name=f"{main_outcome}_mean")
    roi_layer_map["is_best_layer"] = True

    full_df = full_df.merge(
        roi_layer_map, on=["roi", "layer", "metric", "model"], how="left"
        )
    full_df["is_best_layer"] = full_df["is_best_layer"].fillna(False)
    subject_avg_df = subject_avg_df.merge(roi_layer_map,
                         on=["roi", "layer", "metric", "model"],
                         how="left",
                         validate="m:1")
    subject_avg_df["is_best_layer"] = subject_avg_df["is_best_layer"].fillna(False)

    # Get model ordering as the average score across all ROIs
    model_orders = {}
    for metric in ['rsa', 'linear_predictivity']:
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

        for roi in df['roi'].unique():
            model_orders[f"{metric}_{roi}"] = get_order(
                    df,
                    metric=metric,
                    additional_condition = df['roi']==roi,
                    order_by="model",
                    column_name=f"{main_outcome}_mean"
                    )
            df[f"model_order_{metric}_{roi}"] = pd.Categorical(
                df["model"],
                categories=model_orders[f"{metric}_{roi}"],
                ordered=True)
            full_df[f"model_order_{metric}_{roi}"] = pd.Categorical(
                full_df["model"],
                categories=model_orders[f"{metric}_{roi}"],
                ordered=True)
            best_model = model_orders[f"{metric}_{roi}"][-1]
            full_df.loc[full_df.query(f"metric == '{metric}' and roi == '{roi}' and split == 'test'"
                          f" and is_best_layer and model == '{best_model}'").index, "is_best_model"] = True
            subject_avg_df.loc[subject_avg_df.query(f"metric == '{metric}' and roi == '{roi}' and split == 'test'"
                          f" and is_best_layer and model == '{best_model}'").index, "is_best_model"] = True
    subject_avg_df["is_best_model"] = subject_avg_df["is_best_model"].fillna(False)
    full_df["is_best_model"] = full_df["is_best_model"].fillna(False)


    # get best model for each ROI and metric based on APS similarity
    _df, roi_layer_map = select_top_layer(subject_avg_df, split="train")

    for metric in full_df['metric'].unique():
        for roi in full_df['roi'].unique():
            model_order = get_order(
                    _df,
                    metric=metric,
                    column_name="aps_score_mean",
                    additional_condition = _df['roi']==roi,
                    order_by="model"
                    )
            best_model = model_order[-1]
            full_df.loc[full_df.query(f"metric == '{metric}' and roi == '{roi}' and split == 'test'"
                          f" and is_best_layer and model == '{best_model}'").index,
                        "is_best_model_aps"] = True

            subject_avg_df.loc[subject_avg_df.query(f"metric == '{metric}' and roi == '{roi}' and split == 'test'"
                          f" and is_best_layer and model == '{best_model}'").index,
                        "is_best_model_aps"] = True
    full_df["is_best_model_aps"] = full_df["is_best_model_aps"].fillna(False)

    # Save intermediate results


    pickle.dump(intersubject_df, open(SCRATCH / "intersubject_df.pkl", "wb"))
    pickle.dump(pairwise_subject_df, open(SCRATCH / "pairwise_subject_df.pkl", "wb"))
    pickle.dump(full_df, open(SCRATCH / "full_df.pkl", "wb"))

    # Bootstrap analysis
    bootstrap_ci_dict = { # type: ignore[var-annotated]
        metric: {
            roi: {
                model: []
                for model in full_df['model'].unique()} for roi in full_df['roi'].unique()
            } for metric in full_df['metric'].unique()
        }

    bootstrap_ci_dict_aps = { # type: ignore[var-annotated]
        metric: {
            roi: {
                model: []
                for model in full_df['model'].unique()} for roi in full_df['roi'].unique()
            } for metric in full_df['metric'].unique()
        }

    n_bootstrap_samples = 1000
    sampled_subjects_all = []

    for _bootstrap_sample in tqdm(range(n_bootstrap_samples)):
        out, sampled_subjects = bootstrap_rows_by_subject_fast(
            full_df.query("split == 'test' and is_best_layer"),
            random_state=_bootstrap_sample)
        sampled_subjects_all.append(sampled_subjects)
        subject_avg_df_bs = out.groupby(
            ["model", "layer", "roi", "split", "metric", "stream"], as_index=False
        ).agg(
            count=("score", "count"),
            score_mean=(f"{main_outcome}", "mean"),
            score_std=(f"{main_outcome}", "std"),
            aps_score_mean=("similarity", "mean"),
            aps_score_std=("similarity", "std"),
        )

        models = subject_avg_df_bs['model'].unique()
        metrics = subject_avg_df_bs['metric'].unique()


        for _, row in subject_avg_df_bs.iterrows():
            metric = row["metric"]
            roi    = row["roi"]
            model  = row["model"]
            score  = row["score_mean"]
            aps_score = row["aps_score_mean"]
            bootstrap_ci_dict[metric][roi][model].append(score)
            bootstrap_ci_dict_aps[metric][roi][model].append(aps_score)

    bootstrap_ci = {
        metric: {
            roi: {
                model: np.percentile(bootstrap_ci_dict[metric][roi][model], [2.5, 97.5])
                for model in full_df['model'].unique()} for roi in full_df['roi'].unique()
            } for metric in full_df['metric'].unique()
        }
    bootstrap_ci_aps = {
        metric: {
            roi: {
                model: np.percentile(bootstrap_ci_dict_aps[metric][roi][model], [2.5, 97.5])
                for model in full_df['model'].unique()} for roi in full_df['roi'].unique()
            } for metric in full_df['metric'].unique()
        }
    rows = []
    for metric in metrics:
        for roi in rois:
            for model in models:
                rows.append(
                    {
                    "metric": metric,
                    "roi": roi,
                    "model": model,
                    f"{main_outcome}_ci": bootstrap_ci[metric][roi][model],
                    "aps_score_ci": bootstrap_ci_aps[metric][roi][model],
                    "is_best_layer": True,
                })
    ci_df = pd.DataFrame(rows)
    ci_df["is_best_layer"] = ci_df["is_best_layer"].fillna(False)
    subject_avg_df = subject_avg_df.merge(ci_df, on=["roi", "model", "metric",
                                                     "is_best_layer"], how="left")
    logger.info(f"Subject avg df after merging CI: {subject_avg_df.shape}")
    pickle.dump(subject_avg_df, open(SCRATCH / f"subject_avg_df_{main_outcome}.pkl", "wb"))
    subject_avg_df = run_equivalence_analysis(subject_avg_df, main_outcome=main_outcome)


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
    df.insert(
        df.shape[1],
        "model_provider",
        df["model"].apply(lambda x: x.split('/')[0]))

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
    #  add dataset size to subject_avg_df
    ############################################################


    pickle.dump(full_df, open(SCRATCH / "full_df_extended.pkl", "wb"))
    pickle.dump(df, open(SCRATCH / f"df_{main_outcome}.pkl", "wb"))
    pickle.dump(subject_avg_df, open(SCRATCH / f"subject_avg_df_equiv_{main_outcome}.pkl", "wb"))


    with open(SCRATCH / f"bootstrap_ci_{main_outcome}.pkl", "wb") as f:
        pickle.dump(bootstrap_ci, f)

    logger.info("Analysis complete! Results saved to SCRATCH directory.")


if __name__ == "__main__":
    main()

