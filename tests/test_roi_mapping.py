from types import SimpleNamespace

import numpy as np
import pytest

from multitasking.fmri_data.roi_utils import RoiMasks


@pytest.fixture
def mock_hcp(monkeypatch):
    """Simulate a minimal HCP atlas with controlled map and label IDs."""
    # Create fake labels: ID → name
    labels = {1: "L_V1", 181: "R_V1", 2: "L_MT", 182: "R_MT", 3: "L_Pir", 183: "R_Pir"}

    # Create map_all: assign IDs to specific voxel indices
    map_all = np.zeros(20, dtype=int)
    map_all[0:2] = 1  # L_V1
    map_all[2:4] = 181  # R_V1
    map_all[4:6] = 2  # L_MT
    map_all[6:8] = 182  # R_MT
    map_all[8:10] = 3  # L_Pir
    map_all[10:12] = 183  # R_Pir

    mock_mmp = SimpleNamespace(labels=labels, map_all=map_all)
    monkeypatch.setattr(
        "multitasking.fmri_data.roi_utils.hcp", SimpleNamespace(mmp=mock_mmp)
    )


def test_roi_extraction_mapping(mock_hcp):
    roi_names = ["V1", "MT", "Pir"]
    roi_masks = RoiMasks(roi_names)
    roi_masks.extract_rois()

    # Check voxel counts
    assert roi_masks.voxels_per_roi["V1"] == 4
    assert roi_masks.voxels_per_roi["MT"] == 4
    assert roi_masks.voxels_per_roi["Pir"] == 4

    # Check voxel indices
    assert np.array_equal(roi_masks.rois["V1"], np.array([0, 1, 2, 3]))
    assert np.array_equal(roi_masks.rois["MT"], np.array([4, 5, 6, 7]))
    assert np.array_equal(roi_masks.rois["Pir"], np.array([8, 9, 10, 11]))

    # Check index mapping
    assert roi_masks.rois_to_indices["V1"] == (0, 4)
    assert roi_masks.rois_to_indices["MT"] == (4, 8)
    assert roi_masks.rois_to_indices["Pir"] == (8, 12)

    # Check that hcp_roi_mask contains the right IDs at the right spots
    expected_mask = np.zeros(20, dtype=int)
    expected_mask[0:2] = 1
    expected_mask[2:4] = 181
    expected_mask[4:6] = 2
    expected_mask[6:8] = 182
    expected_mask[8:10] = 3
    expected_mask[10:12] = 183

    np.testing.assert_array_equal(roi_masks.hcp_roi_mask, expected_mask)

    # Check atlas ID mapping
    assert roi_masks.roi_to_atlas_ids == {
        "V1": [1, 181],
        "MT": [2, 182],
        "Pir": [3, 183],
    }
