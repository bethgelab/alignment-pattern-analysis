from types import SimpleNamespace

import numpy as np
import pytest

from multitasking.fmri_data.load_fmri import load_dataset
from multitasking.fmri_data.roi_utils import RoiMasks


@pytest.fixture
def mock_hcp(monkeypatch):
    labels = {1: "L_V1", 181: "R_V1", 2: "L_MT", 182: "R_MT", 3: "L_Pir", 183: "R_Pir"}

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


@pytest.mark.parametrize(
    "roi_combo, expected_values",
    [
        (["V1"], [11.0] * 4),
        (["V1", "Pir"], [11.0] * 4 + [33.0] * 4),
        (["V1", "Pir", "MT"], [11.0] * 4 + [33.0] * 4 + [22.0] * 4),
        (["Pir", "V1"], [33.0] * 4 + [11.0] * 4),
    ],
)
def test_load_fmri_with_mock_rois(tmp_path, mock_hcp, roi_combo, expected_values):
    # Step 1: Create synthetic data (shape: n_frames × n_voxels)
    n_frames = 10
    n_voxels = 20
    data = np.zeros((n_frames, n_voxels))

    roi_values = {"V1": 11.0, "Pir": 33.0, "MT": 22.0}
    roi_voxels = {
        "V1": [0, 1, 2, 3],
        "MT": [4, 5, 6, 7],
        "Pir": [8, 9, 10, 11],
    }

    for roi in roi_combo:
        data[:, roi_voxels[roi]] = roi_values[roi]

    # Step 2: Save the data to an .npy file
    fmri_path = tmp_path / "fmri" / "sub-01"
    fmri_path.mkdir(parents=True)
    data_file = fmri_path / "example.npy"
    np.save(data_file, data)

    # Step 3: Create a config.yaml
    config = {
        "fmri": {
            "path": str(tmp_path / "fmri"),
            "sub_id": "sub-01",
            "roi_names": roi_combo,
        }
    }

    # Step 4: Run the pipeline
    roi_masks = RoiMasks(config["fmri"]["roi_names"])
    roi_masks.extract_rois()
    all_data, _ = load_dataset(config, roi_masks)

    # Step 5: Validate
    expected_array = np.array([expected_values] * n_frames)
    np.testing.assert_array_equal(all_data.squeeze(), expected_array)
