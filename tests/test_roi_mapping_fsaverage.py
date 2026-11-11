"""Tests for the fsaverage ROI masks.

Mirrors tests/test_roi_mapping.py, which monkeypatches hcp_utils for the
fsLR32k implementation; here the FreeSurfer annotation reader is patched
instead.
"""

from pathlib import Path

import numpy as np
import pytest

from multitasking.fmri_data import roi_utils_fsaverage
from multitasking.fmri_data.roi_utils_fsaverage import RoiMasksFSAverage

N_VERTS_PER_HEMI = 20

# Colour-table order; index 0 is the unknown/background label, as in the real
# annotation. read_annot returns indices into this list, not atlas label IDs.
LH_NAMES = ["???", "L_V1_ROI", "L_MT_ROI", "L_MST_ROI"]
RH_NAMES = ["???", "R_V1_ROI", "R_MT_ROI", "R_MST_ROI"]


def _fake_labels() -> np.ndarray:
    """Builds labels assigning vertices 0-3 to V1, 4-5 to MT, 6 to MST."""
    labels = np.zeros(N_VERTS_PER_HEMI, dtype=np.int32)
    labels[0:4] = 1  # V1
    labels[4:6] = 2  # MT
    labels[6:7] = 3  # MST
    return labels


@pytest.fixture
def mock_annot(monkeypatch):
    """Patches read_annot so no atlas files are needed."""

    def fake_read_annot(path):
        names = LH_NAMES if Path(path).name.startswith("lh") else RH_NAMES
        return _fake_labels(), None, [name.encode("utf-8") for name in names]

    monkeypatch.setattr(
        roi_utils_fsaverage.nib.freesurfer.io, "read_annot", fake_read_annot
    )
    # Only the annotation files are faked; every other Path.exists is real.
    real_exists = Path.exists
    monkeypatch.setattr(
        Path, "exists", lambda self: self.suffix == ".annot" or real_exists(self)
    )


def test_hemispheres_merge_with_right_offset_and_missing_rois_are_dropped(mock_annot):
    masks = RoiMasksFSAverage(
        ["V1", "NotAnRoi"],
        atlas_dir="/does/not/matter",
        subsample_stride=1,
        strict_rois=False,
    )
    masks.extract_rois()

    # Left vertices 0-3, right vertices 0-3 shifted by the left hemisphere size.
    expected = np.concatenate([np.arange(4), np.arange(4) + N_VERTS_PER_HEMI])
    np.testing.assert_array_equal(masks.rois["V1"], expected)
    # A missing ROI must leave roi_names, or per-ROI lookups would KeyError.
    assert masks.roi_names == ["V1"]
    assert "NotAnRoi" not in masks.voxels_per_roi
    assert "NotAnRoi" not in masks.rois_to_indices


def test_inherited_merge_roi_groups_works(mock_annot):
    masks = RoiMasksFSAverage(
        ["V1", "MT", "MST"],
        {"MT+MST": ["MT", "MST"]},
        atlas_dir="/does/not/matter",
        subsample_stride=1,
    )
    masks.extract_rois()

    assert "MT+MST" in masks.rois
    assert "MT" not in masks.rois and "MST" not in masks.rois
    # MT contributes 2 vertices per hemisphere, MST 1: (2 + 1) * 2 = 6.
    assert masks.voxels_per_roi["MT+MST"] == 6
    assert masks.roi_names == ["V1", "MT+MST"]


