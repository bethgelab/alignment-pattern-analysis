import logging
import pickle
import re
from pathlib import Path

import click
import numpy as np
from benedict import benedict

from multitasking.fmri_data.data_imputation import impute_nans
from multitasking.fmri_data.roi_utils import RoiMasks

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
def load_fmri(config_path: Path) -> tuple[np.ndarray, list[str]]:
    config = benedict.from_yaml(config_path)
    roi_masks = RoiMasks(config["fmri"]["roi_names"])
    roi_masks.extract_rois()
    all_data, all_object_ids = load_dataset(config, roi_masks)
    return all_data, all_object_ids


def load_dataset(
    config: dict, roi_masks: RoiMasks, split: str = "train", z_score: bool = True
) -> tuple[np.ndarray, list[str]]:
    """Load the fMRI data from the path and subject ID specified in config.

    Parameters:
        config: The full configuration dictionary.
        roi_masks: ...
        split: IGNORED for fmri_objaverse dataset. Only used for bold_moments dataset.
    """
    fmri_path = config["fmri"]["path"]
    sub_id = config["fmri"]["sub_id"]

    # check which dataset to load
    if config.get("data.name") == "bold_moments":
        data_path = Path(fmri_path) / sub_id / "prepared_betas"
        if z_score:
            data_path = (
                data_path / f"{sub_id}_organized_betas_task-{split}_normalized.pkl"
            )
        else:
            data_path = (
                data_path / f"{sub_id}_organized_betas_task-{split}_unnormalized.pkl"
            )
        # For RSA, decide later/elsewhere which dataset to use
        with open(data_path, "rb") as f:
            betas = pickle.load(f)
        betas_values = betas[0]

        # IDEA add functionality for different averaging methods
        betas_values = np.nanmean(betas_values, axis=1)
        # add a empty dimension to the array
        betas_values = np.expand_dims(betas_values, axis=1)

        # extract the rois we need
        betas_per_roi = [
            betas_values[:, :, roi_masks.rois[roi_name]]
            for roi_name in roi_masks.roi_names
        ]
        for idx, roi_name in enumerate(roi_masks.roi_names):
            betas_per_roi[idx] = impute_nans(betas_per_roi[idx], roi_name, split)

        betas_values_concat = np.concatenate(
            betas_per_roi,
            axis=2,  # concatenate along the voxel dimension
        )
        betas_values = betas_values_concat

        LOGGER.info("Loaded betas with shape: %s", betas_values.shape)

        betas_video_order = np.array(betas[1])
        # delete "vid" from string

        # delete sample 0258 and 0831 because corrupted
        # also unequal to 1001, 1097, 1090, 1085
        mask = (
            (betas_video_order != "vid0258")
            & (betas_video_order != "vid0831")
            & (betas_video_order != "vid1001")
            & (betas_video_order != "vid1097")
            & (betas_video_order != "vid1090")
            & (betas_video_order != "vid1085")
            & (betas_video_order != "vid1094")
            & (betas_video_order != "vid1093")
        )
        betas_values = betas_values[mask]

        betas_video_ordered = []
        for string in betas_video_order:
            string = string.replace("vid", "")
            if (
                string != "0258"
                and string != "0831"
                and string != "1001"
                and string != "1097"
                and string != "1090"
                and string != "1085"
                and string != "1094"
                and string != "1093"
            ):
                betas_video_ordered.append(string)

        return betas_values, betas_video_ordered

    else:
        parts = [fmri_path, sub_id]
        sub_dir = config["fmri"].get("sub_dir")
        n_frames = 10
        if sub_dir:  # Only add if not None or empty string
            parts.append(sub_dir)
            match = re.search(r"n_frames_(\d+)", sub_dir)
            if match:
                n_frames = int(match.group(1))
        folder = Path(*parts)
        data = []
        object_ids = []
        LOGGER.info("Checking for %s frames", str(n_frames))
        LOGGER.info("Loading .npy files in %s", str(folder))

        for i, file in enumerate(folder.glob("*.npy")):
            tmp = np.load(file)
            if tmp.shape[0] == n_frames:
                all_rois_data = [
                    tmp[:, roi_masks.rois[roi_name]] for roi_name in roi_masks.roi_names
                ]
                data.append(np.hstack(all_rois_data))
                object_ids.append(file.name.split(".")[0])
            if (i % 500) == 0:
                LOGGER.info("%s files screened", str(i))
        LOGGER.info("Done screening, concatenate %s entries...", len(data))
        data = np.stack(data)  # type: ignore
        LOGGER.info(data.shape)  # type: ignore
    return data, object_ids  # type: ignore


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(fmt="%(asctime)s %(name)s %(levelname)s %(message)s")
    load_fmri()
    LOGGER.info("Done")
