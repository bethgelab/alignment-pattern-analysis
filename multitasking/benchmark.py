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

from multitasking.datasets import build_dataset
from multitasking.fmri_data.voxel_consistency import (
    filter_by_voxel_consistency,
    get_roi_wise_voxel_consistency,
)
from multitasking.metrics import RSA, LinearPredictivity
from multitasking.utils.file_handling import config_hash
from multitasking.utils.get_representations import (
    compute_model_representation,
    get_flat_representation,
)
from multitasking.utils.plotting import plot_rdm

LOGGER = logging.getLogger(__name__)

SCRATCH = Path(os.environ.get("SCRATCH", "/scratch"))

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_ROOT = PROJECT_ROOT / "configs"


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CONFIG_ROOT / "benchmark.yaml",
)
@click.option("--model", default="all")
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
    model: str,
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

    base_config = benedict.from_yaml(config_path)

    model_configs = {}
    for path in (CONFIG_ROOT / "models").glob("*.yaml"):
        model_config = benedict.from_yaml(path)
        model_name = model_config["feature_extraction.model"]
        model_configs[model_name] = model_config
    if model != "all":
        model_configs = {model: model_configs[model]}

    for model_index, (model_name, model_config) in enumerate(model_configs.items()):
        LOGGER.info(
            f"Benchmarking model: {model_name} ... "
            f"({model_index + 1} of {len(model_configs)})"
        )
        full_config = base_config.deepcopy()
        full_config.merge(model_config)
        benchmark_model(full_config, output_dir, cache_dir, overwrite, overwrite_cache)


def benchmark_model(
    config: benedict,
    output_dir: Path,
    cache_dir: Path,
    overwrite: bool,
    overwrite_cache: bool,
) -> None:
    """Benchmarks a single model."""
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
    config_inputs_hash = config_hash(config, ["dataset", "feature_extraction", "fmri"])

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

    if dataset_key == "bold_moments":
        dataset_train = build_dataset(config, split="train")
        dataset_test = build_dataset(config, split="test")

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
            ncsnr_per_roi = get_roi_wise_voxel_consistency(
            config["fmri"],
            dataset_train.roi_masks,
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
    """
    if not config.get("rsa.enabled", False):
        LOGGER.info("RSA is disabled. Skipping RSA computation.")
        return

    LOGGER.info("Computing RSA ...")
    rsa = RSA()

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
