import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from multitasking.fmri_data.roi_utils import RoiMasks


def _reformat_data_roi_wise(fmri_data: NDArray[np.float64],
                            roi_masks: RoiMasks):
    """See method of same name in multitasking.datasets, BoldMoments class.

    Reformats data.
    """
    return {roi: fmri_data[..., roi_masks.rois[roi]]
            for roi in roi_masks.roi_names}


def get_roi_wise_voxel_consistency(
    fmri_config: Mapping[str, Any], roi_masks: RoiMasks, subject_id: str,
) -> dict[str, np.ndarray]:
    """Get roi-wise voxel consistency as provided with the original dataset.

    Args:
        fmri_config: configuration for the fMRI data
                    (keys used: path, sub_id, voxel_consistency_threshold)
        roi_masks: ROI masks that the fMRI data is organized by/was created with
        subject_id: the subject id

    Returns:
        dict[str, np.ndarray]: roi-wise voxel consistency
    """
    # Load the noise ceiling data

    ncsnr_path = Path(fmri_config["path"]) / (
        f"{subject_id}/prepared_betas/{subject_id}_noiseceiling_task-train_n-3.pkl"
    )

    with open(ncsnr_path, "rb") as f:
        ncsnr, _ = pickle.load(f)

    # Reformat to per-ROI dict format
    ncsnr_per_roi = _reformat_data_roi_wise(ncsnr, roi_masks)
    return ncsnr_per_roi


def filter_by_voxel_consistency(
    fmri_data: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Apply voxel consistency threshold on fMRI data.

    Args:
        fmri_data (dict[str, np.ndarray]): Dictionary mapping ROI names to fMRI
            data arrays of shape (n_samples, n_voxels) .
        masks: Dictionary mapping ROI names to boolean masks.

    Returns:
        dict[str, np.ndarray]: Dictionary mapping ROI names to filtered fMRI
            data arrays, where only voxels with consistency >= threshold are
            retained.

    Raises:
        ValueError: If ROI names in ncsnr_per_roi and fmri_data do not match.
    """
    if set(masks.keys()) != set(fmri_data.keys()):
        raise ValueError(
            f"ROI names in masks ({set(masks.keys())}) "
            f"do not match ROI names in fmri_data ({set(fmri_data.keys())}), "
            f"did you pass the correct ROI masks?"
        )

    # Filter the fMRI data
    reduced_fmri_data = {}
    for roi in fmri_data.keys():
        reduced_fmri_data[roi] = fmri_data[roi][..., masks[roi]]

    return reduced_fmri_data
