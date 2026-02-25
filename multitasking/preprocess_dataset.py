"""Dataset preprocessing script.

This script preprocesses a video dataset by adapting the spatial resolutions and frame
rates. While not necessary for the benchmarking pipeline, preprocessing will speed up
feature extraction.
"""

import os
import warnings
import zipfile
from pathlib import Path

import click
from executor import execute
from tqdm import tqdm

SCRATCH = Path(os.environ.get("SCRATCH", "/scratch"))


@click.command()
@click.option(
    "--dataset-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to a zip archive containing the videos.",
)
@click.option(
    "--output-path",
    type=click.Path(exists=False, file_okay=False, path_type=Path),
    required=True,
    help=(
        "Path to the output directory. Videos will be saved in this directory, "
        "following the structure of the input archive."
    ),
)
@click.option(
    "--frame-rate",
    type=float,
    default=None,
    help=(
        "Frame rate of the output videos. If not specified, the frame rate of the "
        "input videos will be used."
    ),
)
@click.option(
    "--resolution",
    type=int,
    nargs=2,
    default=None,
    help=(
        "Resolution of the output videos. If not specified, the resolution of the "
        "input videos will be used."
    ),
)
@click.option("--progress/--no-progress", default=True, help="Show progress bar.")
@click.option(
    "--scratch-path",
    type=click.Path(exists=False, file_okay=False, path_type=Path),
    default=SCRATCH / "videos",
    help="Path to the scratch directory.",
)
def preprocess_dataset(
    dataset_path: Path,
    output_path: Path,
    frame_rate: float | None = None,
    resolution: tuple[int, int] | None = None,
    progress: bool = True,
    scratch_path: Path = SCRATCH / "videos",
) -> None:
    """Preprocess a video dataset by adapting the resolution and frame rate."""
    if frame_rate is None and resolution is None:
        warnings.warn(
            "No frame rate or resolution specified. The input videos will not be "
            "resampled or resized.",
            UserWarning,
            stacklevel=2,
        )

    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall(scratch_path)

    videos = sorted(
        video
        for video in scratch_path.glob("**/*.mp4")
        if not video.is_relative_to(scratch_path / "__MACOSX")
    )

    for video in tqdm(videos, desc="Preprocessing...", disable=not progress):
        video_output_path = output_path / video.relative_to(scratch_path)
        video_output_path.parent.mkdir(parents=True, exist_ok=True)

        if frame_rate is None and resolution is None:
            command = f"cp {video} {video_output_path}"

        else:
            filters = []
            if frame_rate is not None:
                filters.append(f"fps={frame_rate}")
            if resolution is not None:
                filters.append(f"scale={resolution[0]}:{resolution[1]}")

            command = (
                f"ffmpeg -y -i {video} -vf {','.join(filters)} {video_output_path}"
            )

        execute(f"{command} > /dev/null 2>&1")


if __name__ == "__main__":
    preprocess_dataset()
