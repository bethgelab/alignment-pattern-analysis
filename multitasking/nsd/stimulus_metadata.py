"""Mapping NSD's shared-1000 stimuli back to their COCO source images.

NSD's stimuli are COCO images cropped to a square and resized to 425x425. The
crop boxes are published in ``nsd_stim_info_merged.csv``, so the stimuli can be
reconstructed from a local COCO copy instead of downloading the 37 GB
``nsd_stimuli.hdf5``.

The chain is:

    shared-1000 stimulus id k (1-based, as used in the prepared betas)
        -> nsd_id = sharedix[k - 1] - 1      (sharedix is 1-based, nsdId is 0-based)
        -> the nsd_stim_info row with that nsdId
        -> coco_id + coco_split + crop box

The off-by-one in that first step is the dangerous part: it would return a
valid but wrong image for every stimulus, with nothing failing. It is checked
by :func:`verify_manifest`, which asserts every mapped row carries the
``shared1000`` flag - true for 1000/1000 rows under the correct reading and
only 12/1000 under the wrong one.
"""

import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

#: Native NSD stimulus resolution, in pixels.
NSD_STIMULUS_RESOLUTION = 425

#: Vendored 1000-row manifest, extracted from the 10.8 MB upstream CSV.
MANIFEST_PATH = Path(__file__).parent / "metadata" / "nsd_shared1000_stimuli.csv"

MANIFEST_COLUMNS = [
    "stimulus_id",
    "nsd_id",
    "coco_id",
    "coco_split",
    "crop_top",
    "crop_bottom",
    "crop_left",
    "crop_right",
]


def load_stim_info(csv_path: str | Path) -> pd.DataFrame:
    """Loads the upstream ``nsd_stim_info_merged.csv``.

    Args:
        csv_path: Path to the downloaded CSV.

    Returns:
        The table, indexed by ``nsdId``.

    Raises:
        FileNotFoundError: If the CSV is missing, naming where to get it.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"NSD stimulus info not found at {csv_path}. Download it from "
            "https://natural-scenes-dataset.s3.amazonaws.com/nsddata/experiments/"
            "nsd/nsd_stim_info_merged.csv"
        )
    return pd.read_csv(csv_path).set_index("nsdId")


def shared1000_manifest(
    expdesign_path: str | Path, stim_info: pd.DataFrame
) -> pd.DataFrame:
    """Builds the shared-1000 stimulus manifest.

    Args:
        expdesign_path: Path to ``nsd_expdesign.mat`` (supplies ``sharedix``).
        stim_info: The table from :func:`load_stim_info`.

    Returns:
        One row per shared-1000 stimulus, with :data:`MANIFEST_COLUMNS`, sorted
        by ``stimulus_id``.

    Raises:
        FileNotFoundError: If the experiment design file is missing.
    """
    from scipy.io import loadmat

    expdesign_path = Path(expdesign_path)
    if not expdesign_path.exists():
        raise FileNotFoundError(
            f"NSD experiment design not found at {expdesign_path}."
        )

    shared_ix = loadmat(expdesign_path)["sharedix"].squeeze()
    # sharedix is 1-based into the 73k stimuli; nsdId is 0-based.
    nsd_ids = shared_ix.astype(np.int64) - 1

    rows = stim_info.loc[nsd_ids]
    crop_boxes = np.array(
        [ast.literal_eval(box) for box in rows["cropBox"]], dtype=np.float64
    )

    return pd.DataFrame(
        {
            "stimulus_id": np.arange(1, len(nsd_ids) + 1, dtype=np.int64),
            "nsd_id": nsd_ids,
            "coco_id": rows["cocoId"].to_numpy(dtype=np.int64),
            "coco_split": rows["cocoSplit"].to_numpy(),
            "crop_top": crop_boxes[:, 0],
            "crop_bottom": crop_boxes[:, 1],
            "crop_left": crop_boxes[:, 2],
            "crop_right": crop_boxes[:, 3],
        },
        columns=MANIFEST_COLUMNS,
    )


def verify_manifest(
    manifest: pd.DataFrame, stim_info: pd.DataFrame | None = None
) -> None:
    """Checks the manifest for the two mistakes that would fail silently.

    Args:
        manifest: The manifest from :func:`shared1000_manifest`.
        stim_info: The upstream table. When given, every mapped row is checked
            to carry the ``shared1000`` flag, which is what catches an
            off-by-one in the sharedix mapping.

    Raises:
        ValueError: If the manifest is the wrong size, its IDs are not the
            full 1..1000 range, or the shared1000 check fails.
    """
    if len(manifest) != 1000:
        raise ValueError(f"Expected 1000 stimuli, got {len(manifest)}")

    expected_ids = list(range(1, 1001))
    if manifest["stimulus_id"].tolist() != expected_ids:
        raise ValueError("stimulus_id must run from 1 to 1000 in order.")

    if manifest["nsd_id"].duplicated().any():
        raise ValueError("Duplicate nsd_id values in the manifest.")

    if stim_info is not None:
        flags = stim_info.loc[manifest["nsd_id"], "shared1000"]
        n_flagged = int(flags.astype(bool).sum())
        if n_flagged != len(manifest):
            raise ValueError(
                f"Only {n_flagged}/{len(manifest)} mapped stimuli carry the "
                "shared1000 flag. This is the signature of an off-by-one in the "
                "sharedix mapping: sharedix is 1-based and nsdId is 0-based, so "
                "the mapping must be nsd_id = sharedix[k - 1] - 1."
            )


def crop_box_to_pixels(
    crop_box: tuple[float, float, float, float], height: int, width: int
) -> tuple[int, int, int, int]:
    """Converts a fractional NSD crop box to pixel coordinates.

    The crop box gives the fraction removed from each side, as
    ``(top, bottom, left, right)``. NSD's boxes always describe a square
    region, but they are not symmetric - the crop was chosen by a saliency
    criterion, not centred - so the side length is computed once and applied to
    both axes rather than rounding each edge independently, which could
    otherwise produce an off-by-one rectangle.

    Args:
        crop_box: ``(top, bottom, left, right)`` fractions removed.
        height: Source image height in pixels.
        width: Source image width in pixels.

    Returns:
        ``(left, upper, right, lower)``, ready for ``PIL.Image.crop``.

    Raises:
        ValueError: If the box does not describe a square region, or falls
            outside the image.
    """
    top, bottom, left, right = crop_box

    kept_height = (1.0 - top - bottom) * height
    kept_width = (1.0 - left - right) * width
    if abs(kept_height - kept_width) > 1.0:
        raise ValueError(
            f"Crop box {crop_box} on a {width}x{height} image keeps a "
            f"{kept_width:.2f}x{kept_height:.2f} region, which is not square."
        )

    side = int(round((kept_height + kept_width) / 2.0))
    upper = int(round(top * height))
    left_px = int(round(left * width))

    # Clamp, so rounding can never run past the image edge.
    upper = max(0, min(upper, height - side))
    left_px = max(0, min(left_px, width - side))

    if side <= 0 or upper + side > height or left_px + side > width:
        raise ValueError(
            f"Crop box {crop_box} does not fit in a {width}x{height} image."
        )

    return left_px, upper, left_px + side, upper + side


def coco_image_path(coco_root: str | Path, coco_split: str, coco_id: int) -> Path:
    """Returns the path of a COCO image in a standard COCO 2017 layout.

    Args:
        coco_root: Directory holding an ``images/`` subdirectory.
        coco_split: For example "train2017".
        coco_id: The COCO image ID.

    Returns:
        The image path.
    """
    return Path(coco_root) / "images" / coco_split / f"{int(coco_id):012d}.jpg"


def load_manifest(path: str | Path | None = None) -> pd.DataFrame:
    """Loads the vendored shared-1000 manifest.

    Args:
        path: Manifest location. Defaults to :data:`MANIFEST_PATH`.

    Returns:
        The manifest table.

    Raises:
        FileNotFoundError: If it is missing, naming the script that builds it.
    """
    path = Path(path) if path is not None else MANIFEST_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"NSD stimulus manifest not found at {path}. Build it with "
            "`python -m multitasking.nsd.prepare_stimulus_manifest`."
        )
    return pd.read_csv(path)
