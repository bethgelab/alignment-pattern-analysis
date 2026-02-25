"""Tests for dataloading."""

from pathlib import Path

import numpy as np
from benedict import benedict

from multitasking.fmri_data.load_fmri import load_dataset
from multitasking.fmri_data.roi_utils import RoiMasks


def _resolve_config_path(relative_path: str) -> Path:
    """Resolve config path whether running from root or tests directory."""
    # Try from root first (most common)
    root_path = Path(relative_path)
    if root_path.exists():
        return root_path
    # Try from tests directory
    tests_path = Path("..") / relative_path
    if tests_path.exists():
        return tests_path
    # Return original path and let it fail with clear error
    return root_path


def test_load_fmri_configs_same_rois_yield_same_data(
    config_path_1: Path = Path("configs/test_configs/test_config_1.yaml"),
    config_path_2: Path = Path("configs/test_configs/test_config_2.yaml"),
):
    config_path_1 = _resolve_config_path(str(config_path_1))
    config_path_2 = _resolve_config_path(str(config_path_2))
    config_1 = benedict.from_yaml(config_path_1)
    roi_masks_1 = RoiMasks(roi_names=config_1["fmri.roi_names"])
    roi_masks_1.extract_rois()
    fmri_data_1, _ = load_dataset(config_1, roi_masks_1)

    config_2 = benedict.from_yaml(config_path_2)
    roi_masks_2 = RoiMasks(roi_names=config_2["fmri.roi_names"])
    roi_masks_2.extract_rois()
    fmri_data_2, _ = load_dataset(config_2, roi_masks_2)

    # otherwise trivial
    assert not (np.all(config_1["fmri.roi_names"] == config_2["fmri.roi_names"]))

    data_same = []
    for roi_name in config_1["fmri.roi_names"]:
        if roi_name in config_2["fmri.roi_names"]:
            data_1 = fmri_data_1[..., slice(*roi_masks_1.rois_to_indices[roi_name])]
            data_2 = fmri_data_2[..., slice(*roi_masks_2.rois_to_indices[roi_name])]
            data_same.append(np.all(data_1 == data_2))
    assert np.all(data_same)
