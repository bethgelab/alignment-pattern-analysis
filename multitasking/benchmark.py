import itertools
import logging
import os
import time
from pathlib import Path
from typing import Any

import click
import joblib
import numpy as np
import pandas as pd
from benedict import benedict

from multitasking.datasets import TEST_SPLIT_CONFIG_KEY, build_dataset
from multitasking.datasets_base import FmriStimulusDataset
from multitasking.fmri_data.voxel_consistency import (
    filter_by_voxel_consistency,
)
from multitasking.metrics import RSA, LinearPredictivity
from multitasking.utils.file_handling import config_hash
from multitasking.utils.get_representations import (
    compute_model_representation,
    get_flat_representation,
)
from multitasking.utils.plotting import plot_rdm

LOGGER = logging.getLogger(__name__)

SUPPORTED_DATASETS = ("bold_moments", "nsd", "nsd_synthetic")

SCRATCH = Path(os.environ.get("SCRATCH", "/scratch"))


def stimulus_set_names(
    config: dict[str, Any],
    dataset_train: FmriStimulusDataset,
    dataset_test: FmriStimulusDataset,
) -> dict[str, str]:
    """Names the stimulus set behind each split, for the scoresheets.

    ``split`` alone is ambiguous once a run can score one dataset's test split
    against another's: "test" means NSD-core's held-out stimuli in one
    scoresheet and NSD-synthetic's 220 in the next, and the analysis layer
    de-duplicates and merges noise ceilings on ``split``.

    Args:
        config: The full configuration dictionary.
        dataset_train: The train split's dataset.
        dataset_test: The test split's dataset.

    Returns:
        ``{split: stimulus set name}``.
    """
    base_name = str(config["dataset"].get("name", "bold_moments"))
    override = config.get(TEST_SPLIT_CONFIG_KEY) or {}
    test_name = str((override.get("dataset") or {}).get("name", base_name))

    def name_of(dataset: FmriStimulusDataset, fallback: str) -> str:
        return str(getattr(dataset, "STIMULUS_SET", "") or fallback)

    return {
        "train": name_of(dataset_train, base_name),
        "test": name_of(dataset_test, test_name),
    }


def check_splits_are_comparable(
    dataset_train: FmriStimulusDataset, dataset_test: FmriStimulusDataset
) -> None:
    """Checks that the two splits share a voxel layout and an input size.

    Differing sample counts are normal - BOLD Moments is 1000/102 and NSD
    618/148 - but the voxel layout must be identical, because the pipeline
    builds the voxel-consistency mask once from the *train* dataset's NCSNR as
    a positional vector in ROI-vertex order and applies it to both splits, and
    because linear predictivity fits on train voxels and predicts test voxels.

    A stride mismatch surfaces as an IndexError and an ROI-set mismatch as a
    KeyError, but an ROI *ordering* mismatch is silent unless
    ``apply_consistency_threshold`` happens to be on. Hence this check.

    Args:
        dataset_train: The train split's dataset.
        dataset_test: The test split's dataset.

    Raises:
        ValueError: If the two disagree on ROI names and their order, on any
            ROI's vertex indices, or on the stimulus resolution.
    """
    train_masks, test_masks = dataset_train.roi_masks, dataset_test.roi_masks

    if list(train_masks.roi_names) != list(test_masks.roi_names):
        raise ValueError(
            "The train and test splits have different ROIs, or the same ROIs "
            f"in a different order: {list(train_masks.roi_names)} vs "
            f"{list(test_masks.roi_names)}. Their voxels would not line up."
        )

    for roi_name in train_masks.roi_names:
        if not np.array_equal(
            train_masks.rois[roi_name], test_masks.rois[roi_name]
        ):
            raise ValueError(
                f"The train and test splits select different vertices for "
                f"{roi_name} ({len(train_masks.rois[roi_name])} vs "
                f"{len(test_masks.rois[roi_name])}). They must share an atlas, "
                "a subsample stride and any ROI grouping."
            )

    train_resolution = getattr(dataset_train, "resolution", None)
    test_resolution = getattr(dataset_test, "resolution", None)
    if train_resolution != test_resolution:
        raise ValueError(
            f"The train and test splits resize stimuli differently "
            f"({train_resolution} vs {test_resolution}), so their features are "
            "not comparable."
        )


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--output-dir",
    "output_dir",
    type=click.Path(exists=False, file_okay=False, path_type=Path),
    default=Path(__file__).parent.parent / "output",
)
@click.option(
    "--cache-dir",
    "cache_dir",
    type=click.Path(exists=False, file_okay=False, path_type=Path),
    default=SCRATCH / "cache",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=True,
    help="Enable or disable overwriting of existing output files.",
)
@click.option(
    "--overwrite-cache/--no-overwrite-cache",
    default=False,
    help="Enable or disable overwriting of existing cache.",
)
def benchmark(
    config_path: Path,
    output_dir: Path,
    cache_dir: Path,
    overwrite: bool,
    overwrite_cache: bool,
):
    """Main entrypoint for the benchmark."""
    if overwrite:
        LOGGER.warning(
            "Overwriting existing output. Disable this by setting --no-overwrite."
        )
    else:
        LOGGER.info("Existing outputs will not be overwritten.")

    if not overwrite_cache:
        LOGGER.warning(
            "Using cached data. Disable the cache for reproducible results by setting "
            "--overwrite-cache.",
        )

    config = benedict.from_yaml(config_path)
    LOGGER.info(config)
    config_str = (
        "model_"
        + config["feature_extraction"]["model"]
        + "_fmridata_"
        + "_".join(config["fmri"].get("sub_id", ["all"]))
    )
    # replace all "/ with _ in the config setting
    config_str = config_str.replace("/", "_")
    if not (overwrite):
        curr_time_str = f"_{time.time()}"
        config["curr_time"] = curr_time_str
    config_full_hash = config_hash(config)
    config_inputs_hash = config_hash(
        config,
        ["dataset", "feature_extraction", "fmri", TEST_SPLIT_CONFIG_KEY],
    )

    output_dir = Path(output_dir) / config_full_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Writing results to {output_dir}")

    debug_dir = Path(output_dir) / "debugging"
    debug_dir.mkdir(exist_ok=True, parents=True)

    # write config_hash and output_dir to the config file
    config["output_dir"] = str(output_dir)
    config["config_hash"] = config_full_hash
    config["config_str"] = config_str
    config["overwrite"] = overwrite
    config["overwrite_cache"] = overwrite_cache
    # Save the config as a YAML file
    with open(output_dir / "config.yaml", "w") as f:
        f.write(config.to_yaml())

    # Create cache folder
    Path(cache_dir).mkdir(exist_ok=True)
    cache_dir = Path(cache_dir) / config_inputs_hash
    cache_dir.mkdir(exist_ok=True, parents=True)
    LOGGER.info(f"Cache at {cache_dir}")

    dataset_key = config["dataset"].get("name", "bold_moments")
    splits = ["train", "test"]

    if dataset_key in SUPPORTED_DATASETS:
        dataset_train = build_dataset(config, split="train")
        dataset_test = build_dataset(config, split="test")
        check_splits_are_comparable(dataset_train, dataset_test)
        stimulus_sets = stimulus_set_names(config, dataset_train, dataset_test)
        LOGGER.info(
            "Train split: %d samples from %s. Test split: %d samples from %s.",
            len(dataset_train.sample_ids),
            stimulus_sets["train"],
            len(dataset_test.sample_ids),
            stimulus_sets["test"],
        )

        reduced_features_train, feature_projectors_train = compute_model_representation(
            config=config,
            dataset=dataset_train,
            cache_path=cache_dir,
            reuse=not (overwrite_cache),
            split="train",
            feature_projectors=None,
        )

        reduced_features_test, _ = compute_model_representation(
            config=config,
            dataset=dataset_test,
            cache_path=cache_dir,
            reuse=not (overwrite_cache),
            split="test",
            feature_projectors=feature_projectors_train,
        )
        split_model_representation = get_flat_representation(
            {
                "train": reduced_features_train,
                "test": reduced_features_test,
            }
        )

        subject_ids = dataset_train.subject_ids

        rois = config["fmri"]["roi_names"]
        layers = list(reduced_features_train.keys())
        LOGGER.info(f"---dev-log--- rois: {rois}")
        LOGGER.info(f"---dev-log--- layers: {layers}")
        LOGGER.info(
            f"Loading fMRI data for {len(dataset_train.subject_ids)} subjects ... "
        )
        fmri_data_train = dataset_train.load_fmri_data()
        fmri_data_test = dataset_test.load_fmri_data()

        LOGGER.info(
            f"Loading fMRI data for {len(dataset_train.subject_ids)} subjects done."
        )

        ncsnr_per_subject_per_roi: dict[str, dict[str, np.ndarray]] = \
            {subject: {} for subject in subject_ids}
        masks_per_subject_per_roi: dict[str, dict[str, np.ndarray]] = \
            {subject: {} for subject in subject_ids}

        threshold = max(0, config["fmri"].get("voxel_consistency_threshold", 0))
        LOGGER.info(f"Creating voxel consistency masks with "
                    f"threshold {threshold}")

        for subject_id in subject_ids:
            ncsnr_per_roi = dataset_train.load_voxel_consistency(
                subject_id
            )
            for roi in fmri_data_train[subject_id].keys():
                mask = ncsnr_per_roi[roi] >= threshold
                masks_per_subject_per_roi[subject_id][roi] = mask

            ncsnr_per_subject_per_roi[subject_id] = ncsnr_per_roi

        if config["fmri"].get("apply_consistency_threshold", False):
            LOGGER.info(
                "Filtering by voxel consistency with threshold "
                f"{config['fmri']['voxel_consistency_threshold']} ..."
            )

            for subject_id in subject_ids:
                fmri_data_train[subject_id] = filter_by_voxel_consistency(
                    fmri_data_train[subject_id],
                    masks_per_subject_per_roi[subject_id]
                )
                fmri_data_test[subject_id] = filter_by_voxel_consistency(
                    fmri_data_test[subject_id],
                    masks_per_subject_per_roi[subject_id]
                )


        for subject_id in subject_ids:
            LOGGER.info(f"Computing alignment metrics for subject {subject_id}")
            split_brain_representation = get_flat_representation(
                {
                    "train": fmri_data_train[subject_id],
                    "test": fmri_data_test[subject_id],
                }
            )

            if config["feature_extraction"].get("shuffle", False):
                # Shuffle the fMRI data wrt the model data to get chance level results
                for roi in rois:
                    split_brain_representation[f"train_{roi}"] = (
                        split_brain_representation[f"train_{roi}"][
                            np.random.permutation(
                                split_brain_representation[f"train_{roi}"].shape[0]
                            )
                        ]
                    )
                    split_brain_representation[f"test_{roi}"] = (
                        split_brain_representation[f"test_{roi}"][
                            np.random.permutation(
                                split_brain_representation[f"test_{roi}"].shape[0]
                            )
                        ]
                    )
                LOGGER.warning("Shuffled fMRI data wrt the model data.")

            compute_rsa(
                config,
                split_model_representation,
                split_brain_representation,
                layers,
                rois,
                splits,
                output_path=output_dir,
                subject_id=subject_id,
                stimulus_sets=stimulus_sets,
            )

            compute_linear_predictivity(
                split_model_representation,
                split_brain_representation,
                layers,
                rois,
                splits,
                config,
                output_path=output_dir,
                subject_id=subject_id,
                stimulus_sets=stimulus_sets,
            )

    else:
        raise ValueError(f"Dataset {dataset_key} not supported!")


def compute_rsa(
    config: dict[str, Any],
    model_representation: dict[str, np.ndarray],
    brain_representation: dict[str, np.ndarray],
    layers: list[str],
    rois: list[str],
    splits: list[str],
    output_path: Path,
    subject_id: str,
    stimulus_sets: dict[str, str] | None = None,
) -> None:
    """Computes the model-brain alignment using RSA.

    This function orchestrates all steps that are necessary to compute the model-brain
    alignment using RSA:

    1. Compute the model RDMs
    2. Compute the brain RDMs
    3. Compute correlations between model RDMs and brain RDMs

    Parameters:
        config: The full configuration dictionary.
        model_representation: The model representation.
        fmri_data: The fMRI data.
        rois_to_indices: The rois to indices.
        cache_path: The path to the cache directory.
        output_path: The path to the output directory.
        stimulus_sets: ``{split: stimulus set name}``, recorded per row so
            that "test" is not ambiguous between datasets. Defaults to the
            configured dataset name for every split.
    """
    if not config.get("rsa.enabled", False):
        LOGGER.info("RSA is disabled. Skipping RSA computation.")
        return

    LOGGER.info("Computing RSA ...")
    rsa = RSA()
    stimulus_sets = stimulus_sets or {}
    default_stimulus_set = str(config["dataset"].get("name", "bold_moments"))

    model_rdms: dict[str, np.ndarray] = {}
    brain_rdms: dict[str, np.ndarray] = {}
    scores: list[dict[str, Any]] = []

    for split in splits:
        for layer in layers:
            for roi in rois:
                features1 = model_representation[f"{split}_{layer}"]
                features2 = brain_representation[f"{split}_{roi}"]

                score, details = rsa(features1, features2)
                if np.isnan(score) and features1.shape[-1] > 0 \
                        and features2.shape[-1] > 0:
                    LOGGER.warning(f"NaN score for {split}_{layer}_{roi}!")

                scores.append(
                    {
                        "model": config["feature_extraction"]["model"],
                        "layer": layer,
                        "roi": roi,
                        "split": split,
                        "stimulus_set": stimulus_sets.get(
                            split, default_stimulus_set
                        ),
                        "subject": subject_id,
                        "n_features_1_model": features1.shape[-1],
                        "n_features_2_brain": features2.shape[-1],
                        "metric": "rsa",
                        "score": score,
                    }
                )

                assert isinstance(details["rdm1"], np.ndarray)
                assert isinstance(details["rdm2"], np.ndarray)
                model_rdms[f"{split}_{layer}"] = details["rdm1"]
                brain_rdms[f"{split}_{roi}"] = details["rdm2"]

    scores_path = output_path / "scoresheets" / f"scores_rsa_{subject_id}.csv"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(scores_path, index=False)
    LOGGER.info(f"Computing RSA completed.\n{scores_df}")
    if config["rsa"].get("visualize_rdms", False):
        LOGGER.info("Visualizing RDMs ...")
        rdm_heatmaps_path = output_path / "rdm_heatmaps"
        rdm_heatmaps_path.mkdir(parents=True, exist_ok=True)

        model_name = config["feature_extraction"]["model"]
        for split, layer in itertools.product(splits, layers):
            model_rdm = model_rdms[f"{split}_{layer}"]
            if not os.path.exists(rdm_heatmaps_path / f"model_rdm_{layer}_{split}.png"):
                plot_rdm(
                    model_rdm,
                    title=f"RDM for model '{model_name}',  "
                    "layer '{layer}' ({split} set)",
                    output_path=rdm_heatmaps_path / f"model_rdm_{layer}_{split}.png",
                )

        for split, roi in itertools.product(splits, rois):
            brain_rdm = brain_rdms[f"{split}_{roi}"]
            plot_rdm(
                brain_rdm,
                title=f"RDM for brain region '{roi}' ({split} set)",
                output_path=rdm_heatmaps_path
                / f"brain_rdm_{roi}_{split}_{subject_id}.png",
            )

        LOGGER.info("Visualizing RDMs completed.")


def compute_linear_predictivity(
    model_representations,
    brain_representations,
    layers,
    rois,
    splits,
    config,
    output_path: Path,
    subject_id: str,
    stimulus_sets: dict[str, str] | None = None,
) -> None:
    if not config.get("linear_predictivity.enabled", False):
        LOGGER.info(
            "Linear predictivity is disabled. Skipping linear predictivity computation."
        )
        return

    LOGGER.info("Computing linear predictivity ...")

    kwargs = {
        key: value
        for key, value in config["linear_predictivity"].items()
        if key != "enabled"
    }
    linear_predictivity = LinearPredictivity(**kwargs)
    stimulus_sets = stimulus_sets or {}
    default_stimulus_set = str(config["dataset"].get("name", "bold_moments"))

    scores: list[dict[str, Any]] = []

    ridgecv_path = output_path / "ridgecv"
    ridgecv_path.mkdir(parents=True, exist_ok=True)

    for layer in layers:
        for roi in rois:
            for split in splits:
                features1 = model_representations[f"{split}_{layer}"]
                features2 = brain_representations[f"{split}_{roi}"]
                if split == "train":
                    #  For train, fit the model
                    score, details = linear_predictivity(features1, features2)
                    train_score = score
                else:
                    #  For test, evaluate the model
                    if np.isnan(score):
                        score = np.nan # e.g. if one featuremap has 0 feature dims
                    else:
                        score = linear_predictivity.evaluate_models( # type: ignore
                                features1, features2
                        )
                    test_scores = score

                scores.append(
                    {
                        "model": config["feature_extraction"]["model"],
                        "layer": layer,
                        "roi": roi,
                        "split": split,
                        "stimulus_set": stimulus_sets.get(
                            split, default_stimulus_set
                        ),
                        "subject": subject_id,
                        "metric": "linear_predictivity",
                        "n_features_1_model": features1.shape[-1],
                        "n_features_2_brain": features2.shape[-1],
                        "score": np.nanmean(score),
                        "test_scores_list": (score if split == "test" else None),
                        "alphas": (details["alphas"] if split == "train" else None),

                        "train_set_train_split_score_mean":
                            (np.mean(details["train_scores"]).item() if \
                                split == "train" else None),
                        "ridgecv_val_mse":
                            (np.mean(details["ridgecv_val_mse"]).item() \
                                if split == "train" else None),
                        "train_set_train_split_mse":
                            (np.mean(details["train_mse"]).item() \
                                if split == "train" else None),
                        "train_set_val_split_mse":
                            (np.mean(details["test_mse"]).item() \
                                if split == "train" else None),

                    }
                )

            # Bundle model + metadata
            package = {
                "alphas": details["alphas"],
            }
            ridgecv_fpath = (ridgecv_path /
                            f"alphas_{subject_id}_{roi}_{layer}.pkl")
            joblib.dump(package, ridgecv_fpath)

            LOGGER.info(
                f"{roi}, {layer}, alphas: {details['alphas']},"
                f" train score: {train_score:.2f}, test_score: "
                f"{np.nanmean(test_scores):.2f}"
            )
    scores_path = (
        output_path / "scoresheets" / f"scores_linear_predictivity_{subject_id}.csv"
    )
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(scores_path, index=False)

    LOGGER.info(f"Computing linear predictivity completed.\n{scores_df}")



if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(fmt="%(asctime)s %(name)s %(levelname)s %(message)s")
    benchmark()
