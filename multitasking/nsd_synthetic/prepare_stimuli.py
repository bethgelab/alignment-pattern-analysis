"""Extracts the shared NSD-synthetic stimuli into per-image PNGs.

Unlike NSD-core, whose stimuli are COCO crops that this repository rebuilds
from a local COCO copy, NSD-synthetic's images are generated and ship as
pixels in ``nsdsynthetic_stimuli.hdf5``. So there is no reconstruction to get
wrong here - but two framing decisions are, and both are recorded in
``provenance.json``:

*Only the 220 achromatic images are written.* The other 64 are chromatic and
ship as eight per-subject renderings, one per subject's monitor calibration.
Model features are extracted once per split and reused for every subject, so
per-subject stimuli would break the sample-ordering contract.

"Achromatic" is NSD-synthetic's own term and means "no chromatic modulation",
not ``R == G == B``: every pixel of these 220 lies on a single ray through RGB
space, along the display's white point of (252, 220, 216), so they carry a
uniform warm cast. That is a property of the stimuli as presented, and is
preserved rather than normalised away.

*The 1360x714 frame is cropped to its central 714x714.* NSD-synthetic spans
16 degrees by 8.4 of visual field where NSD-core spans 8.4 by 8.4. Cropping to
the centre matches NSD-core's field of view, so features extracted from the two
datasets are comparable - at the cost of each stimulus's left and right
periphery.

Usage:
    python -m multitasking.nsd_synthetic.prepare_stimuli \
        --stimuli-hdf5 path/to/nsdsynthetic_stimuli.hdf5 \
        --image-info path/to/nsdsyntheticimageinformation.csv \
        --output-dir path/to/nsd_synthetic_stimuli
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

from multitasking.nsd_synthetic.design import (
    NSD_SYNTHETIC_N_ACHROMATIC,
    NSD_SYNTHETIC_N_IMAGES,
    NSD_SYNTHETIC_STIMULUS_SET,
    format_sample_id,
)

LOGGER = logging.getLogger(__name__)

IMAGES_DIRNAME = "images"
MANIFEST_FILENAME = "manifest.csv"
PROVENANCE_FILENAME = "provenance.json"

#: Dataset holding the images inside the HDF5, as in NSD-core's stimulus file.
IMG_BRICK_KEY = "imgBrick"

#: Published pixel size of an NSD-synthetic stimulus, 16 by 8.4 degrees.
NSD_SYNTHETIC_SOURCE_SIZE = (714, 1360)

#: Side length of the central square, 8.4 by 8.4 degrees, matching NSD-core.
NSD_SYNTHETIC_CROPPED_SIZE = 714

#: Stimulus IDs derived from natural NSD scenes: four scenes, their upside-down,
#: Mooney and line-drawing versions, and a second set of four scenes. Flagged
#: because an NSD-core train split may contain the same underlying images, which
#: would mean the encoding model has seen them. The released synthetic metadata
#: carries no COCO identifiers, so this cannot be resolved from the data here.
SCENE_DERIVED_STIMULUS_IDS = tuple(range(13, 33))

#: Columns of nsdsyntheticimageinformation.csv, renamed to snake case.
IMAGE_INFO_COLUMNS = {
    "Image number": "stimulus_id",
    "Image": "image",
    "Image subclass number": "image_subclass_number",
    "Image subclass": "image_subclass",
    "Image class number": "image_class_number",
    "Image class": "image_class",
}


def load_image_info(path: Path) -> pd.DataFrame:
    """Loads and validates the per-image class metadata.

    Args:
        path: Path to ``nsdsyntheticimageinformation.csv``.

    Returns:
        A frame indexed by position with snake-case columns, one row per image.

    Raises:
        ValueError: If the file does not describe all 284 images in order.
    """
    # The published file carries a UTF-8 BOM.
    info = pd.read_csv(path, encoding="utf-8-sig")

    missing = [name for name in IMAGE_INFO_COLUMNS if name not in info.columns]
    if missing:
        raise ValueError(
            f"{path} is missing columns {missing}; found {list(info.columns)}."
        )
    info = info[list(IMAGE_INFO_COLUMNS)].rename(columns=IMAGE_INFO_COLUMNS)

    if len(info) != NSD_SYNTHETIC_N_IMAGES:
        raise ValueError(
            f"{path} describes {len(info)} images, expected "
            f"{NSD_SYNTHETIC_N_IMAGES}."
        )
    expected_ids = np.arange(1, NSD_SYNTHETIC_N_IMAGES + 1)
    if not np.array_equal(info["stimulus_id"].to_numpy(), expected_ids):
        raise ValueError(
            f"{path} is not in contiguous 1..{NSD_SYNTHETIC_N_IMAGES} order."
        )
    return info


def load_achromatic_images(path: Path) -> NDArray[np.uint8]:
    """Reads the shared achromatic stimuli out of the HDF5.

    Args:
        path: Path to ``nsdsynthetic_stimuli.hdf5``.

    Returns:
        An array of shape ``(220, 714, 1360, 3)`` and dtype uint8.

    Raises:
        ImportError: If h5py is not installed.
        ValueError: If the file's contents are not what this design implies.
    """
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "Reading nsdsynthetic_stimuli.hdf5 needs h5py, which is not "
            "installed. Install it with `uv sync` or `pip install h5py`."
        ) from error

    with h5py.File(path, "r") as source:
        if IMG_BRICK_KEY not in source:
            raise ValueError(
                f"{path} has no {IMG_BRICK_KEY!r} dataset; found "
                f"{list(source)}."
            )
        images = np.asarray(source[IMG_BRICK_KEY][:])

    if images.shape[0] != NSD_SYNTHETIC_N_ACHROMATIC:
        raise ValueError(
            f"{path} holds {images.shape[0]} images, expected "
            f"{NSD_SYNTHETIC_N_ACHROMATIC}. The chromatic images live in the "
            "per-subject nsdsynthetic_colorstimuli_subjNN.hdf5 files and are "
            "deliberately not used here."
        )
    if images.dtype != np.uint8:
        raise ValueError(f"{path} holds {images.dtype} images, expected uint8.")

    # NSD-core's imgBrick is (N, H, W, 3); be explicit rather than trusting it,
    # since a channels-first file would silently transpose every stimulus.
    if images.ndim != 4:
        raise ValueError(f"{path} holds {images.ndim}-D data, expected 4-D.")
    if images.shape[1:] == (*NSD_SYNTHETIC_SOURCE_SIZE, 3):
        pass
    elif images.shape[1:] == (3, *NSD_SYNTHETIC_SOURCE_SIZE):
        images = images.transpose(0, 2, 3, 1)
    else:
        raise ValueError(
            f"{path} holds images of shape {images.shape[1:]}, expected "
            f"{(*NSD_SYNTHETIC_SOURCE_SIZE, 3)} or "
            f"{(3, *NSD_SYNTHETIC_SOURCE_SIZE)}."
        )
    return images


def crop_and_resize(
    image: NDArray[np.uint8], resolution: int = NSD_SYNTHETIC_CROPPED_SIZE
) -> Image.Image:
    """Crops one stimulus to its central square and resizes it.

    Args:
        image: One stimulus of shape ``(714, 1360, 3)``.
        resolution: Output side length in pixels.

    Returns:
        A square RGB image.
    """
    height, width = image.shape[:2]
    side = min(height, width)
    left = (width - side) // 2
    cropped = image[:, left : left + side]

    stimulus = Image.fromarray(cropped, mode="RGB")
    if stimulus.size != (resolution, resolution):
        stimulus = stimulus.resize(
            (resolution, resolution), Image.Resampling.LANCZOS
        )
    return stimulus


@click.command()
@click.option(
    "--stimuli-hdf5",
    "stimuli_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to nsdsynthetic_stimuli.hdf5.",
)
@click.option(
    "--image-info",
    "image_info_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to nsdsyntheticimageinformation.csv.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory to write images/, manifest.csv and provenance.json into.",
)
@click.option(
    "--resolution",
    default=NSD_SYNTHETIC_CROPPED_SIZE,
    show_default=True,
    help="Output side length in pixels.",
)
@click.option(
    "--exclude-id",
    "excluded_ids",
    multiple=True,
    type=int,
    help=(
        "Stimulus IDs to leave out, repeatable. Intended for the "
        "scene-derived images, should they turn out to overlap the NSD-core "
        "train split."
    ),
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Rewrite images that already exist.",
)
def prepare_stimuli(
    stimuli_path: Path,
    image_info_path: Path,
    output_dir: Path,
    resolution: int,
    excluded_ids: tuple[int, ...],
    overwrite: bool,
) -> None:
    """Writes the shared achromatic NSD-synthetic stimuli into output_dir."""
    info = load_image_info(image_info_path)
    images = load_achromatic_images(stimuli_path)
    LOGGER.info(
        "Read %d achromatic stimuli of shape %s from %s",
        len(images),
        images.shape[1:],
        stimuli_path,
    )

    manifest = info.iloc[:NSD_SYNTHETIC_N_ACHROMATIC].copy()
    excluded = sorted(set(excluded_ids))
    if excluded:
        unknown = [i for i in excluded if not 1 <= i <= NSD_SYNTHETIC_N_ACHROMATIC]
        if unknown:
            raise ValueError(
                f"Cannot exclude {unknown}: the achromatic stimuli are "
                f"1..{NSD_SYNTHETIC_N_ACHROMATIC}."
            )
        keep = ~manifest["stimulus_id"].isin(excluded)
        manifest = manifest[keep]
        images = images[keep.to_numpy()]
        LOGGER.warning("Excluding %d stimuli: %s", len(excluded), excluded)

    manifest.insert(
        1, "sample_id", [format_sample_id(i) for i in manifest["stimulus_id"]]
    )

    LOGGER.warning(
        "Stimuli %s are derived from natural NSD scenes. The released "
        "NSD-synthetic metadata carries no COCO identifiers, so whether they "
        "overlap an NSD-core train split cannot be checked here. At most %d of "
        "%d stimuli; use --exclude-id to drop them.",
        list(SCENE_DERIVED_STIMULUS_IDS),
        len(SCENE_DERIVED_STIMULUS_IDS),
        NSD_SYNTHETIC_N_ACHROMATIC,
    )

    images_dir = output_dir / IMAGES_DIRNAME
    images_dir.mkdir(parents=True, exist_ok=True)

    checksums, widths, heights = [], [], []
    rows = list(manifest.itertuples())
    for image, row in tqdm(
        zip(images, rows, strict=True), total=len(rows), desc="Writing stimuli"
    ):
        target = images_dir / f"{row.sample_id}.png"
        if overwrite or not target.exists():
            crop_and_resize(image, resolution=resolution).save(
                target, format="PNG"
            )

        with Image.open(target) as written:
            widths.append(written.size[0])
            heights.append(written.size[1])
        checksums.append(hashlib.sha256(target.read_bytes()).hexdigest())

    if set(widths) != {resolution} or set(heights) != {resolution}:
        raise ValueError(
            f"Expected every stimulus to be {resolution}x{resolution}, got "
            f"widths {sorted(set(widths))} and heights {sorted(set(heights))}."
        )

    manifest["sha256"] = checksums
    manifest["width"] = widths
    manifest["height"] = heights
    manifest.to_csv(output_dir / MANIFEST_FILENAME, index=False)

    provenance = {
        "created": datetime.now(timezone.utc).isoformat(),
        "stimulus_set": NSD_SYNTHETIC_STIMULUS_SET,
        "source_hdf5": str(stimuli_path),
        "source_sha256": hashlib.sha256(stimuli_path.read_bytes()).hexdigest(),
        "n_stimuli": len(manifest),
        "excluded_stimulus_ids": excluded,
        "resolution": resolution,
        "source_size": list(NSD_SYNTHETIC_SOURCE_SIZE),
        "crop": (
            "central square of the 1360x714 frame, i.e. 8.4 by 8.4 degrees of "
            "the published 16 by 8.4, matching NSD-core's field of view"
        ),
        "resample_filter": "PIL.Image.Resampling.LANCZOS",
        "chromatic_images": (
            "Stimuli 221-284 are omitted: they ship as eight per-subject "
            "renderings, and model features are extracted once per split."
        ),
        "colour": (
            "Luminance-only: every pixel lies on a single ray through RGB "
            "space along the display white point (252, 220, 216), so the "
            "images are monochrome but not R == G == B. Preserved as "
            "presented."
        ),
        "scene_derived_stimulus_ids": list(SCENE_DERIVED_STIMULUS_IDS),
        "numpy": np.__version__,
        "pillow": Image.__version__,
    }
    with open(output_dir / PROVENANCE_FILENAME, "w") as handle:
        json.dump(provenance, handle, indent=2)

    LOGGER.info("Wrote %d stimuli to %s", len(manifest), images_dir)


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(fmt="%(asctime)s %(name)s %(levelname)s %(message)s")
    prepare_stimuli()
