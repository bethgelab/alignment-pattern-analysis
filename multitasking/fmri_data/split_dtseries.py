import logging
import math
from pathlib import Path

import click
import nibabel as nib
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input-dir",
    "input_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("/mnt/lustre/work/bethge/bkr578/data/fMRI-Objaverse-CIFTI/"),
)
@click.option(
    "--output-dir",
    "output_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("/mnt/lustre/work/bethge/bkr578/data/multitasking_objaverse/fmri_v2/"),
)
@click.option(
    "--metadata-dir",
    "metadata_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("/mnt/lustre/work/bethge/bkr578/data/fMRI-Objaverse/annotations/"),
)
@click.option(
    "--subject-ids",
    "subject_ids",
    type=click.Choice(["0001", "0006", "0007"]),
    multiple=True,
    default=["0001"],
)
@click.option("--delay", "delay", type=float, required=True)
@click.option("--n-frames", "n_frames", type=int, default=10)
def split_dtseries(
    input_dir: Path,
    output_dir: Path,
    metadata_dir: Path,
    subject_ids: list[str],
    delay: float,
    n_frames: int,
) -> None:
    fs = 0.8  # fMRI repetition rate
    n_runs = 53
    for subject_id in subject_ids:
        output_dir_sub = (
            output_dir
            / Path(f"sub-{subject_id}")
            / Path(f"delay_{str(delay).replace('.', '_')}_n_frames_{n_frames}")
        )
        LOGGER.info(
            f"Creating directory {output_dir_sub} "
            f"for subject {subject_id} and delay {delay}"
        )
        output_dir_sub.mkdir(parents=True, exist_ok=False)
        input_dir_sub = input_dir / f"sub-{subject_id}"
        metadata_dir_sub = metadata_dir / f"sub_{subject_id}_beh/"
        for run_id in range(1, n_runs + 1):
            # find and load the fMRI data for subject subject_id and the run run_id
            nii_file = f"sub-{subject_id}_task-shape_run-{run_id}_space-fsLR_den-91k_bold.dtseries.nii"  # noqa: E501
            dtseries_img = nib.load(input_dir_sub / nii_file)
            dtseries_data = dtseries_img.get_fdata()  # type: ignore
            # find and load the corresponding experimental metadata
            df = pd.read_csv(metadata_dir_sub / f"obj_{subject_id[-1]}_{run_id}.csv")
            video_start_time = df["videoStartTime"].values
            stim_path = df["stim_path"].values
            for i in range(len(video_start_time) - 1):
                # class_name = stim_path[i].split("/")[-2]
                video_name = stim_path[i].split("/")[-1].replace("mp4", "npy")
                # convert stimulus time [s] to a frame number
                start_time = math.ceil((video_start_time[i] + delay) / fs)
                # each stimulus presentation corresponds to 10 fMRI frames
                end_time = start_time + n_frames
                save_path = output_dir_sub / f"{video_name}"
                np.save(save_path, dtseries_data[start_time - 1 : end_time - 1])


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(fmt="%(asctime)s %(name)s %(levelname)s %(message)s")
    split_dtseries()
