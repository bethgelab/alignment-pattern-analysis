"""Tests for mapping NSD stimuli back to their COCO source images."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multitasking.nsd.stimulus_metadata import (
    MANIFEST_PATH,
    coco_image_path,
    crop_box_to_pixels,
    load_manifest,
    shared1000_manifest,
    verify_manifest,
)

EXPDESIGN_PATH = Path(
    "/mnt/lustre/work/bethge/bkr857/projects/metrics_benchmark/nsd_expdesign.mat"
)
COCO_ROOT = Path("/mnt/lustre/datasets/coco")


def _fake_stim_info(n_rows: int = 10) -> pd.DataFrame:
    """Builds a stim_info table where only even nsdIds are shared stimuli."""
    return pd.DataFrame(
        {
            "nsdId": np.arange(n_rows),
            "cocoId": np.arange(n_rows) + 1000,
            "cocoSplit": ["train2017"] * n_rows,
            "cropBox": ["(0, 0, 0.1, 0.1)"] * n_rows,
            "shared1000": [i % 2 == 0 for i in range(n_rows)],
        }
    ).set_index("nsdId")


# --- crop box arithmetic ----------------------------------------------------


def test_crop_box_removes_the_given_fractions():
    # 20% off each side of a 100x100 image leaves a centred 60x60 square.
    box = crop_box_to_pixels((0.2, 0.2, 0.2, 0.2), height=100, width=100)
    assert box == (20, 20, 80, 80)


def test_crop_box_handles_asymmetric_crops():
    """NSD's crops are saliency-chosen, so they are frequently off-centre."""
    # A 640x425 image cropped to a 425x425 square, offset to the right.
    left_fraction = 0.3
    right_fraction = 1.0 - 425 / 640 - left_fraction
    box = crop_box_to_pixels(
        (0.0, 0.0, left_fraction, right_fraction), height=425, width=640
    )
    left, upper, right, lower = box
    assert right - left == lower - upper == 425
    assert left == 192


def test_crop_box_output_is_always_square():
    rng = np.random.default_rng(0)
    for _ in range(50):
        height = int(rng.integers(200, 640))
        width = int(rng.integers(200, 640))
        side = min(height, width)
        top_px = int(rng.integers(0, height - side + 1))
        left_px = int(rng.integers(0, width - side + 1))
        crop = (
            top_px / height,
            (height - side - top_px) / height,
            left_px / width,
            (width - side - left_px) / width,
        )
        left, upper, right, lower = crop_box_to_pixels(crop, height, width)
        assert right - left == lower - upper
        assert 0 <= left and right <= width
        assert 0 <= upper and lower <= height


def test_non_square_crop_box_raises():
    # Keeps 80% of the height but only 20% of the width: not square.
    with pytest.raises(ValueError, match="not square"):
        crop_box_to_pixels((0.1, 0.1, 0.4, 0.4), height=100, width=100)


def test_coco_image_path_uses_twelve_digit_names():
    path = coco_image_path("/coco", "train2017", 262145)
    assert path == Path("/coco/images/train2017/000000262145.jpg")


# --- the off-by-one ---------------------------------------------------------


def test_shared1000_flag_discriminates_the_two_readings():
    """The wrong reading must fail: sharedix is 1-based, nsdId is 0-based.

    This is the property verify_manifest relies on: on real data the correct
    reading flags 1000/1000 rows as shared1000, the off-by-one only 12/1000.
    """
    stim_info = _fake_stim_info(10)  # only even nsdIds are shared stimuli
    correct_ids = [0, 2, 4, 6, 8]
    off_by_one_ids = [i + 1 for i in correct_ids]

    assert stim_info.loc[correct_ids, "shared1000"].all()
    assert not stim_info.loc[off_by_one_ids, "shared1000"].any()


def test_verify_manifest_catches_wrong_size():
    manifest = pd.DataFrame({"stimulus_id": [1, 2], "nsd_id": [0, 1]})
    with pytest.raises(ValueError, match="Expected 1000"):
        verify_manifest(manifest)


def test_verify_manifest_catches_wrong_id_range():
    manifest = pd.DataFrame(
        {"stimulus_id": list(range(0, 1000)), "nsd_id": list(range(1000))}
    )
    with pytest.raises(ValueError, match="1 to 1000"):
        verify_manifest(manifest)


def test_verify_manifest_catches_duplicate_nsd_ids():
    manifest = pd.DataFrame(
        {"stimulus_id": list(range(1, 1001)), "nsd_id": [0] * 1000}
    )
    with pytest.raises(ValueError, match="Duplicate"):
        verify_manifest(manifest)


def test_verify_manifest_catches_unflagged_rows():
    stim_info = _fake_stim_info(2000)
    manifest = pd.DataFrame(
        {
            "stimulus_id": list(range(1, 1001)),
            # Odd nsdIds are not shared stimuli in the fake table.
            "nsd_id": list(range(1, 2001, 2)),
        }
    )
    with pytest.raises(ValueError, match="shared1000 flag"):
        verify_manifest(manifest, stim_info)


# --- the vendored manifest --------------------------------------------------


def test_vendored_manifest_is_present_and_valid():
    manifest = load_manifest()
    verify_manifest(manifest)
    assert len(manifest) == 1000
    assert manifest["coco_split"].isin({"train2017", "val2017"}).all()


def test_missing_manifest_raises_with_guidance(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_stimulus_manifest"):
        load_manifest(tmp_path / "absent.csv")


@pytest.mark.skipif(
    not EXPDESIGN_PATH.exists(), reason="nsd_expdesign.mat not available"
)
def test_vendored_manifest_matches_fresh_mapping():
    """The checked-in manifest must not drift from the experiment design."""
    from scipy.io import loadmat

    shared_ix = loadmat(EXPDESIGN_PATH)["sharedix"].squeeze()
    manifest = load_manifest()
    np.testing.assert_array_equal(
        manifest["nsd_id"].to_numpy(), shared_ix.astype(np.int64) - 1
    )


@pytest.mark.skipif(not COCO_ROOT.exists(), reason="COCO not available")
def test_every_source_image_resolves():
    manifest = load_manifest()
    missing = [
        row.stimulus_id
        for row in manifest.itertuples()
        if not coco_image_path(COCO_ROOT, row.coco_split, row.coco_id).exists()
    ]
    assert not missing, f"{len(missing)} source images missing, e.g. {missing[:5]}"


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="manifest not built")
def test_manifest_crop_boxes_describe_square_regions():
    """The invariant that catches a misread crop box, checked on real data."""
    pytest.importorskip("PIL")
    from PIL import Image

    if not COCO_ROOT.exists():
        pytest.skip("COCO not available")

    manifest = load_manifest()
    for row in manifest.itertuples():
        path = coco_image_path(COCO_ROOT, row.coco_split, row.coco_id)
        with Image.open(path) as image:
            width, height = image.size
        left, upper, right, lower = crop_box_to_pixels(
            (row.crop_top, row.crop_bottom, row.crop_left, row.crop_right),
            height,
            width,
        )
        assert right - left == lower - upper


@pytest.mark.skipif(
    not EXPDESIGN_PATH.exists(), reason="nsd_expdesign.mat not available"
)
def test_shared1000_manifest_round_trip():
    stim_info = _fake_stim_info(80000)
    manifest = shared1000_manifest(EXPDESIGN_PATH, stim_info)
    assert len(manifest) == 1000
    assert manifest["stimulus_id"].tolist() == list(range(1, 1001))
