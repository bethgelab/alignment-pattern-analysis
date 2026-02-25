import hcp_utils as hcp
import numpy as np
import pytest

from multitasking.fmri_data.load_fmri import load_dataset
from multitasking.fmri_data.roi_utils import RoiMasks


@pytest.mark.parametrize("roi_combo", [["V1"], ["V1", "Pir"], ["V1", "Pir", "MT"]])
def test_multi_roi_extraction(tmp_path, roi_combo):
    atlas = hcp.mmp
    labels = atlas.labels
    map_all = atlas.map_all  # ~91282 long

    # 1. ROI names and values we'll inject
    roi_values = {"V1": 11.0, "Pir": 22.0, "MT": 33.0}

    # 2. Build a voxel index map: roi_name → voxel indices
    roi_voxel_indices = {}
    for roi_name in roi_values:
        roi_names_atlas = [f"L_{roi_name}", f"R_{roi_name}"]
        roi_ids = [id_ for id_, name in labels.items() if name in roi_names_atlas]
        indices = np.where(np.isin(map_all, roi_ids))[0]
        assert indices.size > 0, f"No voxels found for ROI {roi_name}"
        roi_voxel_indices[roi_name] = indices

    # 3. Create fake data with unique values per ROI
    n_frames = 5
    n_voxels = map_all.shape[0]
    data = np.zeros((n_frames, n_voxels))

    for roi in roi_combo:
        data[:, roi_voxel_indices[roi]] = roi_values[roi]

    # 4. Save .npy file in standard format
    fmri_dir = tmp_path / "fmri" / "sub01" / f"n_frames_{n_frames}"
    fmri_dir.mkdir(parents=True)
    np.save(fmri_dir / "roi_test.npy", data)

    # 5. Create config.yaml with selected ROIs
    config = {
        "fmri": {
            "roi_names": roi_combo,
            "path": str(tmp_path / "fmri"),
            "sub_id": "sub01",
            "sub_dir": f"n_frames_{n_frames}",
        }
    }

    # 6. Load data
    roi_masks = RoiMasks(config["fmri"]["roi_names"])
    roi_masks.extract_rois()
    all_data, object_ids = load_dataset(config, roi_masks)

    # 7. Check result
    assert object_ids == ["roi_test"]
    assert all_data.shape[0] == 1  # one file
    assert all_data.shape[1] == n_frames

    # Total expected ROI voxels
    expected_voxels = np.concatenate([roi_voxel_indices[roi] for roi in roi_combo])
    assert all_data.shape[2] == len(expected_voxels)

    # 8. Verify values: we can slice and check
    # Stack the expected values in order of ROI_combo
    expected_data = np.hstack(
        [
            np.full((n_frames, len(roi_voxel_indices[roi])), roi_values[roi])
            for roi in roi_combo
        ]
    )

    np.testing.assert_array_equal(all_data[0], expected_data)
