"""Builds the vendored shared-1000 stimulus manifest.

The upstream nsd_stim_info_merged.csv is 10.8 MB and covers all 73k NSD
stimuli; only the 1000 shared ones are needed here, so this extracts them into
a small checked-in manifest.

Usage:
    python -m multitasking.nsd.prepare_stimulus_manifest \
        --stim-info nsd_stim_info_merged.csv --expdesign nsd_expdesign.mat
"""

import logging
from pathlib import Path

import click

from multitasking.nsd.stimulus_metadata import (
    MANIFEST_PATH,
    load_stim_info,
    shared1000_manifest,
    verify_manifest,
)

LOGGER = logging.getLogger(__name__)

STIM_INFO_URL = (
    "https://natural-scenes-dataset.s3.amazonaws.com/nsddata/experiments/nsd/"
    "nsd_stim_info_merged.csv"
)


@click.command()
@click.option(
    "--stim-info",
    "stim_info_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help=f"Path to nsd_stim_info_merged.csv (download: {STIM_INFO_URL}).",
)
@click.option(
    "--expdesign",
    "expdesign_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to nsd_expdesign.mat.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=MANIFEST_PATH,
    show_default=True,
    help="Where to write the manifest.",
)
def prepare_stimulus_manifest(
    stim_info_path: Path, expdesign_path: Path, output_path: Path
) -> None:
    """Extracts the shared-1000 rows and writes the manifest."""
    stim_info = load_stim_info(stim_info_path)
    manifest = shared1000_manifest(expdesign_path, stim_info)
    verify_manifest(manifest, stim_info)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)

    LOGGER.info(
        "Wrote %d stimuli to %s (COCO splits: %s)",
        len(manifest),
        output_path,
        sorted(manifest["coco_split"].unique()),
    )


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(fmt="%(asctime)s %(name)s %(levelname)s %(message)s")
    prepare_stimulus_manifest()
