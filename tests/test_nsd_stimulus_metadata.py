"""Tests for mapping NSD stimuli back to their COCO source images."""

import numpy as np
import pandas as pd
import pytest

from multitasking.nsd.stimulus_metadata import crop_box_to_pixels, verify_manifest


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


def test_crop_box_is_square_and_off_centre_crops_are_handled():
    """NSD's crops are saliency-chosen, so they are frequently off-centre."""
    # A 640x425 image cropped to a 425x425 square, offset to the right.
    left_fraction = 0.3
    right_fraction = 1.0 - 425 / 640 - left_fraction
    left, upper, right, lower = crop_box_to_pixels(
        (0.0, 0.0, left_fraction, right_fraction), height=425, width=640
    )
    assert right - left == lower - upper == 425
    assert left == 192

    # Keeps 80% of the height but only 20% of the width: not square.
    with pytest.raises(ValueError, match="not square"):
        crop_box_to_pixels((0.1, 0.1, 0.4, 0.4), height=100, width=100)


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


