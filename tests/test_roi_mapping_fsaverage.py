"""Tests for the fsaverage ROI masks.

Mirrors tests/test_roi_mapping.py, which monkeypatches hcp_utils for the
fsLR32k implementation; here the FreeSurfer annotation reader is patched
instead.
"""

from pathlib import Path

import numpy as np
import pytest

from multitasking.fmri_data import roi_utils_fsaverage
from multitasking.fmri_data.roi_utils import RoiMasks, build_roi_masks
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
    monkeypatch.setattr(Path, "exists", lambda self: True)


def _build(rois, mock_annot_unused=None, **kwargs):
    masks = RoiMasksFSAverage(rois, atlas_dir="/does/not/matter", **kwargs)
    masks.extract_rois()
    return masks


def test_both_hemispheres_are_merged_with_right_offset(mock_annot):
    masks = _build(["V1"], subsample_stride=1)
    # Left vertices 0-3, right vertices 0-3 shifted by the left hemisphere size.
    expected = np.concatenate([np.arange(4), np.arange(4) + N_VERTS_PER_HEMI])
    np.testing.assert_array_equal(masks.rois["V1"], expected)
    assert masks.voxels_per_roi["V1"] == 8


def test_subsample_stride_decimates_each_hemisphere(mock_annot):
    masks = _build(["V1"], subsample_stride=2)
    # Every second of the four vertices per hemisphere: 0, 2 and 20, 22.
    expected = np.array([0, 2, N_VERTS_PER_HEMI + 0, N_VERTS_PER_HEMI + 2])
    np.testing.assert_array_equal(masks.rois["V1"], expected)


def test_rois_to_indices_are_contiguous_and_in_roi_names_order(mock_annot):
    masks = _build(["V1", "MT", "MST"], subsample_stride=1)
    assert list(masks.rois_to_indices) == masks.roi_names

    expected_start = 0
    for roi_name in masks.roi_names:
        start, end = masks.rois_to_indices[roi_name]
        assert start == expected_start
        assert end - start == masks.voxels_per_roi[roi_name]
        expected_start = end


def test_missing_roi_raises_when_strict(mock_annot):
    with pytest.raises(ValueError, match="not found"):
        _build(["V1", "NotAnRoi"], subsample_stride=1)


def test_missing_roi_is_dropped_from_roi_names_when_not_strict(mock_annot):
    masks = _build(["V1", "NotAnRoi"], subsample_stride=1, strict_rois=False)
    # It must leave roi_names, or downstream per-ROI lookups would KeyError.
    assert masks.roi_names == ["V1"]
    assert "NotAnRoi" not in masks.rois
    assert "NotAnRoi" not in masks.voxels_per_roi
    assert "NotAnRoi" not in masks.rois_to_indices


def test_all_rois_missing_raises(mock_annot):
    with pytest.raises(ValueError):
        _build(["NotAnRoi"], subsample_stride=1, strict_rois=False)


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


def test_roi_mask_marks_selected_vertices(mock_annot):
    masks = _build(["V1"], subsample_stride=1)
    assert masks.hcp_roi_mask.shape == (2 * N_VERTS_PER_HEMI,)
    assert masks.hcp_roi_mask.sum() == 8


def test_inherited_map_rois_to_voxels(mock_annot):
    masks = _build(["V1"], subsample_stride=1)
    mapped = masks.map_rois_to_voxels(np.arange(8, dtype=np.float64))
    assert mapped.shape == (2 * N_VERTS_PER_HEMI,)
    np.testing.assert_array_equal(mapped[masks.rois["V1"]], np.arange(8))


def test_surface_space_markers_are_declarative():
    # isinstance cannot discriminate these, since one subclasses the other.
    assert RoiMasks.surface_space == "fsLR32k"
    assert RoiMasksFSAverage.surface_space == "fsaverage"
    assert issubclass(RoiMasksFSAverage, RoiMasks)


def test_atlas_dir_is_required():
    with pytest.raises(ValueError, match="atlas_dir"):
        RoiMasksFSAverage(["V1"])


@pytest.mark.parametrize("stride", [0, -1, 1.5])
def test_invalid_subsample_stride_raises(stride):
    with pytest.raises(ValueError, match="subsample_stride"):
        RoiMasksFSAverage(["V1"], atlas_dir="/x", subsample_stride=stride)


def test_missing_annotation_file_raises_with_guidance(tmp_path):
    masks = RoiMasksFSAverage(["V1"], atlas_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="HCP-MMP1.annot"):
        masks.extract_rois()


def test_factory_returns_fslr32k_by_default():
    masks = build_roi_masks(["V1"])
    assert type(masks) is RoiMasks
    assert masks.surface_space == "fsLR32k"


def test_factory_returns_fsaverage_when_requested():
    masks = build_roi_masks(
        ["V1"], surface_space="fsaverage", atlas_dir="/x", subsample_stride=6
    )
    assert isinstance(masks, RoiMasksFSAverage)
    assert masks.subsample_stride == 6


def test_factory_rejects_unknown_surface_space():
    with pytest.raises(ValueError, match="Unknown surface_space"):
        build_roi_masks(["V1"], surface_space="mni152")
