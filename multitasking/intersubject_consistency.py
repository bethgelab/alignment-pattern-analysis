"""Intersubject consistency script.

Similar to benchmark.py, but for intersubject consistency.

Differences in config (compared to benchmark.py):
------------------------------------------------
(The config isusually created in launch.py; check out
tasks/bold_moments_benchmark_intersubject_consistency/launch.py)
    fmri:
        intersubject (new!):
            mode: "pairwise" | "leave-one-out-mean"
                    Not used except for consistency checks (that
                     the run is doing what it's supposed to do).
            target_sub_ids: list[str]:
                    The target subjects to compute the intersubject
                    consistency for. We loop over those.
            source_sub_ids: list[list[str]] | None:
                    If None, we compute the leave-one-out-mean of
                    all subjects except the target subject.
                    Else expecting one list of source subjects per
                    target; we average over those.
                    (One source -> no averaging.)
    feature_extraction:
        model: specifies the source subject(s) instead of a model.
        layers / other feature_extraction settings: ignored

"""
import logging
import os
from pathlib import Path

import click
import numpy as np
from benedict import benedict

from multitasking.benchmark import (
    compute_anshksoni_metric,
    compute_linear_predictivity,
    compute_rsa,
    compute_versa,
)
from multitasking.compute_procrustes import compute_procrustes
from multitasking.datasets import build_dataset
from multitasking.fmri_data.voxel_consistency import (
    filter_by_voxel_consistency,
    get_roi_wise_voxel_consistency,
)
from multitasking.utils.file_handling import config_hash
from multitasking.utils.get_representations import (
    get_flat_representation,
)

LOGGER = logging.getLogger(__name__)

SCRATCH = Path(os.environ.get("SCRATCH", "/scratch"))


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
    "--overwrite/--no-overwrite",
    default=True,
    help="Enable or disable overwriting of existing output files.",
)
def intersubject_consistency(
    config_path: Path,
    output_dir: Path,
    overwrite: bool
):
    """Main entrypoint for intersubject consistency computation."""
    if overwrite:
        LOGGER.warning(
            "Overwriting existing output. Disable this by setting --no-overwrite."
        )
    else:
        LOGGER.info("Existing outputs will not be overwritten.")

    config = benedict.from_yaml(config_path)
    LOGGER.info(config)
    config_str = (
        "intersubject_reliability"
        + "_fmridata_"
        + "_".join(config["fmri"].get("sub_id", ["all"]))
    )
    # replace all "/ with _ in the config setting
    config_str = config_str.replace("/", "_")
    config_full_hash = config_hash(config)

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

    # Consistency checks for pairwise intersubject runs
    _mode = config["fmri"]["intersubject"]["mode"]
    if _mode == "pairwise":
        # if not isinstance(config["fmri"]["intersubject"]["source_sub_ids"], str):
        if "source_sub_ids" not in config["fmri"]["intersubject"]:
            raise ValueError( \
                "source_sub_ids required for pairwise intersubject consistency.")
        if not isinstance(config["fmri"]["intersubject"]["source_sub_ids"], list):
            raise ValueError("source_sub_ids must be a list")
        for source_subs in config["fmri"]["intersubject"]["source_sub_ids"]:
            if not isinstance(source_subs, list) or len(source_subs) != 1:
                raise ValueError("source_sub_ids must be a list of lists, with "
                    f"lenght 1 of each sub-list. Found sub-list: {source_subs}")
    # / consistency check end


    # Save the config as a YAML file
    with open(output_dir / "config.yaml", "w") as f:
        f.write(config.to_yaml())

    dataset_key = config["dataset"].get("name", "bold_moments")
    splits = ["train", "test"]

    if dataset_key == "bold_moments":
        dataset_train = build_dataset(config, split="train")
        dataset_test = build_dataset(config, split="test")

        subject_ids = dataset_train.subject_ids

        rois = config["fmri"]["roi_names"]
        LOGGER.info(f"---dev-log--- rois: {rois}")
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

        assert "intersubject" in config["fmri"], "intersubject configuration required."
        if "target_sub_ids" not in config["fmri"]["intersubject"]:
            raise ValueError("target_sub_ids required for intersubject consistency."
                             "This gives the target subjects of the current run. "
                             "Usually set in launch.py."
                             "Set to null or 'all'to compute intersubject consistency "
                             "for all subjects.")
        else:
            subs_for_intersubject = \
                config["fmri"]["intersubject"]["target_sub_ids"]
            if (subs_for_intersubject is None) or (subs_for_intersubject == "all"):
                subs_for_intersubject = subject_ids
            assert isinstance(subs_for_intersubject, list), (
                    "intersubject.target_sub_ids must be a list, null/None or 'all'"
            )
            assert np.all([sub in subject_ids for sub in subs_for_intersubject]), (
                    f"Subjects for intersubject consistency must be in {subject_ids}, "
                    "else their data is not loaded."
            )
            if "source_sub_ids" in config["fmri"]["intersubject"]:
                source_sub_ids_all = config["fmri"]["intersubject"]["source_sub_ids"]
            else:
#<<<<<<< anshksoni_metrics
#                assert np.all([sub in subject_ids for sub in subs_for_intersubject]), (
#                    f"Subjects for intersubject consistency must be in {subject_ids}"
#                )
#        else:
#            subs_for_intersubject = subject_ids
#=======
                source_sub_ids_all = None
#>>>>>>> main

        for idx, subject_id in enumerate(subs_for_intersubject):
            LOGGER.info(f"Computing intersubject alignment metrics "
                        f"for target subject {subject_id}")

            # Get source subject(s) if given (None means all other subjects)
            if source_sub_ids_all is not None:
                source_sub_ids = source_sub_ids_all[idx]
            else:
                source_sub_ids = None

            # Compute leave-one-out-mean of remaining subjects, or
            # specific subject average if source_sub_ids is not None.
            loo_mean_fmri_train = compute_mean(fmri_data_train,
                                               subject_id,
                                               source_sub_ids)
            loo_mean_fmri_test = compute_mean(fmri_data_test,
                                              subject_id,
                                              source_sub_ids)

            if config["fmri"].get("apply_consistency_threshold", False):
                LOGGER.info(
                        "Filtering by voxel consistency with threshold "
                        f"{config['fmri']['voxel_consistency_threshold']} ..."
                )
                # Filter subject by voxel consistency
                fmri_data_train_filtered = filter_by_voxel_consistency(
                    fmri_data_train[subject_id],
                    masks_per_subject_per_roi[subject_id]
                )
                fmri_data_test_filtered = filter_by_voxel_consistency(
                    fmri_data_test[subject_id],
                    masks_per_subject_per_roi[subject_id]
                )

                # Filter leave-one-out-mean by voxel consistency
                mean_ncsnr_per_roi = compute_mean(
                    ncsnr_per_subject_per_roi,
                    subject_id,
                    source_sub_ids)
                mask = {roi: mean_ncsnr_per_roi[roi] >= threshold
                        for roi in mean_ncsnr_per_roi.keys()}
                loo_mean_fmri_train = filter_by_voxel_consistency(
                    loo_mean_fmri_train,
                    mask
                )
                loo_mean_fmri_test = filter_by_voxel_consistency(
                    loo_mean_fmri_test,
                    mask
                )
            else:
                fmri_data_train_filtered = fmri_data_train[subject_id]
                fmri_data_test_filtered = fmri_data_test[subject_id]

            split_brain_representation = get_flat_representation(
                {
                    "train": fmri_data_train_filtered,
                    "test": fmri_data_test_filtered,
                }
            )
            split_mean_representation = get_flat_representation(
                {
                    "train": loo_mean_fmri_train,
                    "test": loo_mean_fmri_test,
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

            n_frames = 1

            compute_rsa(
                config,
                split_mean_representation,
                split_brain_representation,
                rois,
                rois,
                splits,
                output_path=output_dir,
                subject_id=subject_id,
            )

            compute_versa(
                split_mean_representation,
                split_brain_representation,
                rois,
                rois,
                splits,
                config,
                output_path=output_dir,
                subject_id=subject_id,
            )

            compute_procrustes(
                split_mean_representation,
                split_brain_representation,
                rois,
                rois,
                splits,
                config,
                overwrite_score=config["overwrite"],
                n_frames=n_frames,
                use_cv_ankhsoni_prokrustes=config["procrustes"].get(
                    "use_cv_ankhsoni_prokrustes",
                    False,
                ),
            )

            compute_linear_predictivity(
                split_mean_representation,
                split_brain_representation,
                rois,
                rois,
                splits,
                config,
                output_path=output_dir,
                subject_id=subject_id,
            )

            for metric_name in config.get("anshksoni_metrics", {}).keys():
                compute_anshksoni_metric(
                    metric_name,
                    split_mean_representation,
                    split_brain_representation,
                    rois,
                    rois,
                    splits,
                    config,
                    output_path=output_dir,
                    subject_id=subject_id,
                )

    else:
        raise ValueError(f"Dataset {dataset_key} not supported!")


def compute_mean(data: dict[str, dict[str, np.ndarray]],
                 subject_id: str,
                 subs_to_average: list[str] | None = None
                 ) -> dict[str, np.ndarray]:
    """Switch between leave-one-out mean computation and averaging over subs_to_average.

    Parameters:
    -----------
    data: dict[str, dict[str, np.ndarray]]
        The data to compute the mean of.
        Format: {subject_id: {roi: data}}
    subject_id: str
        The left-out subject.
    subs_to_average: list[str] | None
        The subjects to average over. If None, compute the leave-one-out mean
        without subject subject_id. Must not contain subject_id.

    Returns:
    -----------
    dict[str, np.ndarray]
        The mean of the data.
        Format: {roi: data}
    """
    if subs_to_average is None:
        return compute_leave_one_out_mean(data, subject_id)
    else:
        if subject_id in subs_to_average:
            raise ValueError(f"Subject {subject_id} (target subject) "
                            "should not be in subs_to_average.")
        return compute_mean_over_subs(data, subs_to_average)


def compute_mean_over_subs(data: dict[str, dict[str, np.ndarray]],
                            subs_to_average: list[str]) -> dict[str, np.ndarray]:
    """Compute the mean of the data over the given subjects."""
    mean_per_roi = {}
    if len(subs_to_average) == 0:
        raise ValueError("subs_to_average is empty.")
    sub_0 = subs_to_average[0]
    for roi in data[sub_0].keys():
        mean_per_roi[roi] = np.zeros_like(data[sub_0][roi])

    N = len(subs_to_average)
    roi_names = list(data[sub_0].keys())
    for roi in roi_names:
        for sub in subs_to_average:
            mean_per_roi[roi] += data[sub][roi]
        mean_per_roi[roi] /= N

    return mean_per_roi


def compute_leave_one_out_mean(data: dict[str, dict[str, np.ndarray]],
                               subject_id: str) -> dict[str, np.ndarray]:
    """Compute the leave-one-out-mean of the data, leaving out sujbect subject_id.

    Args:
        data: dict[str, dict[str, np.ndarray]]
            The data to compute the leave-one-out-mean of.
            Format: {subject_id: {roi: data}}
        subject_id: str
            The subject to leave out. Must be in data.keys()

    Returns:
        dict[str, np.ndarray]
            The leave-one-out-mean of the data.
            Format: {roi: data}
    """
    loo_mean_per_roi = {}
    for roi in data[subject_id].keys():
        loo_mean_per_roi[roi] = np.zeros_like(data[subject_id][roi])

    count = 0
    for subject, roi_data_dict in data.items():
        if subject == subject_id:
            continue
        count += 1
        for roi, data_dict in roi_data_dict.items():
            if np.any(np.isnan(data_dict)):
                LOGGER.warning(f"Found nan in {roi} for subject {subject}; "
                               "imputing zeros for now.")
                # dirty fix, set nan to 0
                data_dict[np.isnan(data_dict)] = 0
            loo_mean_per_roi[roi] += data_dict

    assert count == len(data.keys()) - 1, ("Count of subjects is not equal"
                                           " to the number of subjects minus one")

    for roi, _ in loo_mean_per_roi.items():
        loo_mean_per_roi[roi] /= count

    return loo_mean_per_roi


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(fmt="%(asctime)s %(name)s %(levelname)s %(message)s")
    intersubject_consistency()
