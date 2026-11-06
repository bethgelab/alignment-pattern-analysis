"""Reconstructs NSD's shared-1000 stimuli from a local COCO copy.

NSD's stimuli are COCO images cropped to a square and resized to 425x425, and
the crop boxes are published, so the 1000 shared stimuli can be rebuilt without
downloading the 37 GB ``nsd_stimuli.hdf5``.

This is the one step in the NSD pipeline where a mistake would be invisible: a
wrong ID mapping or a misread crop box yields perfectly plausible images that
are simply the wrong pixels, and every downstream number would be quietly
wrong. All of the checks therefore run *before* any image is written.

Usage:
    python -m multitasking.nsd.prepare_stimuli \
        --coco-root /mnt/lustre/datasets/coco --output-dir path/to/nsd_stimuli
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from multitasking.nsd.stimulus_metadata import (
    NSD_STIMULUS_RESOLUTION,
    coco_image_path,
    crop_box_to_pixels,
    load_manifest,
    verify_manifest,
)

LOGGER = logging.getLogger(__name__)

IMAGES_DIRNAME = "images"
MANIFEST_FILENAME = "manifest.csv"
PROVENANCE_FILENAME = "provenance.json"


def check_sources_exist(manifest: pd.DataFrame, coco_root: Path) -> None:
    """Checks that every source image is present before writing anything.

    Args:
        manifest: The shared-1000 manifest.
        coco_root: Directory holding ``images/<split>/``.

    Raises:
        FileNotFoundError: If any source image is missing.
    """
    missing = [
        str(coco_image_path(coco_root, row.coco_split, row.coco_id))
        for row in manifest.itertuples()
        if not coco_image_path(coco_root, row.coco_split, row.coco_id).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(manifest)} COCO source images are missing "
            f"under {coco_root}, e.g. {missing[:5]}"
        )


def check_crop_boxes(manifest: pd.DataFrame, coco_root: Path) -> None:
    """Checks every crop box against its image before writing anything.

    NSD's crop boxes are not symmetric - the crop was chosen by a saliency
    criterion - but the region they describe is always square. That squareness
    is the invariant worth asserting: a misparsed column order, or confusing
    "fraction removed" with "fraction kept", breaks it on nearly every row.

    Args:
        manifest: The shared-1000 manifest.
        coco_root: Directory holding ``images/<split>/``.

    Raises:
        ValueError: If any crop box does not describe a square region.
    """
    failures = []
    for row in tqdm(
        manifest.itertuples(), total=len(manifest), desc="Checking crop boxes"
    ):
        path = coco_image_path(coco_root, row.coco_split, row.coco_id)
        with Image.open(path) as image:
            width, height = image.size
        try:
            crop_box_to_pixels(
                (row.crop_top, row.crop_bottom, row.crop_left, row.crop_right),
                height,
                width,
            )
        except ValueError as error:
            failures.append(f"stimulus {row.stimulus_id}: {error}")

    if failures:
        raise ValueError(
            f"{len(failures)} of {len(manifest)} crop boxes are not square, "
            f"e.g.:\n  " + "\n  ".join(failures[:5])
        )


def reconstruct_stimulus(
    source_path: Path,
    crop_box: tuple[float, float, float, float],
    resolution: int = NSD_STIMULUS_RESOLUTION,
) -> Image.Image:
    """Rebuilds one NSD stimulus from its COCO source image.

    Args:
        source_path: The COCO image.
        crop_box: ``(top, bottom, left, right)`` fractions removed.
        resolution: Output side length in pixels.

    Returns:
        The cropped and resized RGB image.
    """
    with Image.open(source_path) as source:
        rgb = source.convert("RGB")
    width, height = rgb.size
    box = crop_box_to_pixels(crop_box, height, width)
    return rgb.crop(box).resize((resolution, resolution), Image.Resampling.LANCZOS)


@click.command()
@click.option(
    "--coco-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("/mnt/lustre/datasets/coco"),
    show_default=True,
    help="COCO root, containing images/<split>/.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Where to write images/ and the manifest.",
)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Manifest to use. Defaults to the vendored one.",
)
@click.option(
    "--resolution",
    default=NSD_STIMULUS_RESOLUTION,
    show_default=True,
    help="Output side length in pixels.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Rewrite images that already exist.",
)
def prepare_stimuli(
    coco_root: Path,
    output_dir: Path,
    manifest_path: Path | None,
    resolution: int,
    overwrite: bool,
) -> None:
    """Reconstructs the shared-1000 stimuli into output_dir."""
    manifest = load_manifest(manifest_path)
    verify_manifest(manifest)
    LOGGER.info("Manifest: %d stimuli", len(manifest))

    # Everything that can be checked up front is checked up front.
    check_sources_exist(manifest, coco_root)
    check_crop_boxes(manifest, coco_root)
    LOGGER.info("All source images resolve and all crop boxes are square.")

    images_dir = output_dir / IMAGES_DIRNAME
    images_dir.mkdir(parents=True, exist_ok=True)

    checksums, widths, heights = [], [], []
    for row in tqdm(
        manifest.itertuples(), total=len(manifest), desc="Reconstructing stimuli"
    ):
        target = images_dir / f"{row.stimulus_id:04d}.png"
        if overwrite or not target.exists():
            stimulus = reconstruct_stimulus(
                coco_image_path(coco_root, row.coco_split, row.coco_id),
                (row.crop_top, row.crop_bottom, row.crop_left, row.crop_right),
                resolution=resolution,
            )
            stimulus.save(target, format="PNG")

        with Image.open(target) as written:
            widths.append(written.size[0])
            heights.append(written.size[1])
        checksums.append(hashlib.sha256(target.read_bytes()).hexdigest())

    if set(widths) != {resolution} or set(heights) != {resolution}:
        raise ValueError(
            f"Expected every stimulus to be {resolution}x{resolution}, got "
            f"widths {sorted(set(widths))} and heights {sorted(set(heights))}."
        )

    output_manifest = manifest.copy()
    output_manifest["sha256"] = checksums
    output_manifest["width"] = widths
    output_manifest["height"] = heights
    output_manifest.to_csv(output_dir / MANIFEST_FILENAME, index=False)

    provenance = {
        "created": datetime.now(timezone.utc).isoformat(),
        "coco_root": str(coco_root),
        "resolution": resolution,
        "n_stimuli": len(manifest),
        "resample_filter": "PIL.Image.Resampling.LANCZOS",
        "crop_box_convention": "(top, bottom, left, right) fractions removed",
        "numpy": np.__version__,
        "pillow": Image.__version__,
        # NSD's own pipeline resized with MATLAB imresize, so reconstructions
        # differ from the published stimuli in the least significant bits.
        # Irrelevant at the resolution models are fed, but worth recording.
        "note": "Reconstructed from COCO; not bit-identical to nsd_stimuli.hdf5.",
    }
    with open(output_dir / PROVENANCE_FILENAME, "w") as handle:
        json.dump(provenance, handle, indent=2)

    LOGGER.info("Wrote %d stimuli to %s", len(manifest), images_dir)


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(fmt="%(asctime)s %(name)s %(levelname)s %(message)s")
    prepare_stimuli()
