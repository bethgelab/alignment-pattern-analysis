import logging
import pickle
from pathlib import Path

import coloredlogs
import numpy as np
import pytest
from benedict import benedict

from multitasking.datasets import build_dataset
from multitasking.fmri_data.roi_utils import RoiMasks
from multitasking.fmri_data.voxel_consistency import (
    _reformat_data_roi_wise,
    filter_by_voxel_consistency,
)

LOGGER = logging.getLogger(__name__)

def _write_noiseceiling_pickle(
    base_path: Path, sub_id: str, split: str, ncsnr: np.ndarray
):
    n = 3 if split == "train" else 10
    target = (
        base_path
        / f"{sub_id}/prepared_betas/{sub_id}_noiseceiling_task-{split}_n-{n}.pkl"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    explainable_variance = np.zeros_like(ncsnr)
    with open(target, "wb") as f:
        pickle.dump((ncsnr, explainable_variance), f)


def _make_roi_masks(
    roi_names: list[str], roi_to_indices: dict[str, np.ndarray]
) -> RoiMasks:
    roi_masks = RoiMasks(roi_names)
    # Avoid calling extract_rois; directly set the attributes used
    # by the code under test
    roi_masks.roi_names = roi_names
    roi_masks.rois = roi_to_indices  # type: ignore[attr-defined]
    return roi_masks


def _make_fmri_dict(
    roi_names: list[str],
    voxels_per_roi: dict[str, int],
    n_samples: int = 5,
    n_reps: int | None = 2,
) -> dict[str, np.ndarray]:
    fmri = {}
    for roi in roi_names:
        v = voxels_per_roi[roi]
        if n_reps is None:
            fmri[roi] = np.random.randn(n_samples, v)
        else:
            fmri[roi] = np.random.randn(n_samples, n_reps, v)
    return fmri


def _last_dim(a: np.ndarray) -> int:
    return a.shape[-1]


def _load_and_create_masks(
    fmri_config: dict, roi_masks: RoiMasks, split: str
) -> dict[str, np.ndarray]:
    """Load noise ceiling and create boolean masks for voxel filtering."""
    # Load noise ceiling pickle
    base_path = Path(fmri_config["path"])
    sub_id = fmri_config["sub_id"]
    n = 3 if split == "train" else 10
    ncsnr_path = (
        base_path
        / f"{sub_id}/prepared_betas/{sub_id}_noiseceiling_task-{split}_n-{n}.pkl"
    )

    with open(ncsnr_path, "rb") as f:
        ncsnr, _ = pickle.load(f)

    # Reformat to per-ROI
    ncsnr_per_roi = _reformat_data_roi_wise(ncsnr, roi_masks)

    # Create boolean masks based on threshold
    threshold = fmri_config["voxel_consistency_threshold"]
    masks = {roi: ncsnr_per_roi[roi] >= threshold for roi in ncsnr_per_roi.keys()}

    return masks


@pytest.mark.parametrize("use_reps", [False, True])
def test_filter_applies_threshold_per_voxel_and_roi(tmp_path: Path, use_reps: bool):
    # Setup synthetic fmri data with two ROIs and known voxel-wise consistency values
    roi_names = ["V1", "MT"]
    roi_to_indices = {
        "V1": np.array([0, 1, 2, 3]),
        "MT": np.array([4, 5, 6]),
    }
    ncsnr = np.zeros(7)
    # Make only a subset pass threshold within each ROI
    # V1: pass voxels 1 and 3; MT: pass voxels 5 only
    ncsnr[[1, 3, 5]] = 0.8
    threshold = 0.5

    roi_masks = _make_roi_masks(roi_names, roi_to_indices)
    voxels_per_roi = {"V1": 4, "MT": 3}
    fmri_data = _make_fmri_dict(
        roi_names, voxels_per_roi, n_samples=6, n_reps=(2 if use_reps else None)
    )

    sub_id = "sub-01"
    fmri_config = {
        "path": str(tmp_path),
        "sub_id": sub_id,
        "voxel_consistency_threshold": threshold,
    }
    _write_noiseceiling_pickle(tmp_path, sub_id, split="train", ncsnr=ncsnr)

    # Create masks using the helper
    masks = _load_and_create_masks(fmri_config, roi_masks, split="train")

    filtered = filter_by_voxel_consistency(
        fmri_data=fmri_data,
        masks=masks,
    )

    # Expect last-dimension filtering only; counts reflect mask above
    assert set(filtered.keys()) == set(roi_names)
    assert _last_dim(filtered["V1"]) == 2  # voxels 1 and 3
    assert _last_dim(filtered["MT"]) == 1  # voxel 5
    # Sample and rep dimensions remain unchanged
    assert filtered["V1"].shape[0] == fmri_data["V1"].shape[0]
    if use_reps:
        assert filtered["V1"].shape[1] == fmri_data["V1"].shape[1]


def test_raises_on_roi_mismatch(tmp_path: Path):
    # ncsnr provided for V1 and MT, fmri data only has V1
    roi_names = ["V1", "MT"]
    roi_masks = _make_roi_masks(
        roi_names, {"V1": np.array([0, 1]), "MT": np.array([2, 3])}
    )
    fmri_data = {"V1": np.random.randn(4, 2)}

    fmri_config = {
        "path": str(tmp_path),
        "sub_id": "sub-01",
        "voxel_consistency_threshold": 0.0,
    }
    _write_noiseceiling_pickle(tmp_path, "sub-01", split="train", ncsnr=np.ones(4))

    # Create masks using the helper
    masks = _load_and_create_masks(fmri_config, roi_masks, split="train")

    with pytest.raises(ValueError):
        filter_by_voxel_consistency(fmri_data, masks)


def test_uses_correct_noiseceiling_file_per_split(tmp_path: Path):
    # Prepare different files for train (n=3) and test (n=10) with different masks
    roi_names = ["V1"]
    roi_masks = _make_roi_masks(roi_names, {"V1": np.array([0, 1, 2, 3])})
    fmri_data = {"V1": np.random.randn(5, 4)}
    sub_id = "sub-01"
    fmri_config = {
        "path": str(tmp_path),
        "sub_id": sub_id,
        "voxel_consistency_threshold": 0.5,
    }

    ncsnr_train = np.array([0.9, 0.1, 0.9, 0.1])  # expect 2 voxels
    ncsnr_test = np.array([0.4, 0.4, 0.4, 0.4])  # expect 0 voxels
    _write_noiseceiling_pickle(tmp_path, sub_id, split="train", ncsnr=ncsnr_train)
    _write_noiseceiling_pickle(tmp_path, sub_id, split="test", ncsnr=ncsnr_test)

    # Create masks for train split
    masks_train = _load_and_create_masks(fmri_config, roi_masks, split="train")
    filtered_train = filter_by_voxel_consistency(fmri_data, masks_train)
    assert _last_dim(filtered_train["V1"]) == 2

    # Create masks for test split
    masks_test = _load_and_create_masks(fmri_config, roi_masks, split="test")
    filtered_test = filter_by_voxel_consistency(fmri_data, masks_test)
    assert _last_dim(filtered_test["V1"]) == 0


def test_integration_on_bold_moments_if_available():
    # Use existing test config; skip if paths are unavailable
    # Resolve path whether running from root or tests directory
    config_path_str = "configs/test_configs/test_config_bold_moments_pipeline.yaml"
    config_path = Path(config_path_str)
    if not config_path.exists():
        config_path = Path("..") / config_path_str
    if not config_path.exists():
        pytest.skip("Test config not available")

    config = benedict.from_yaml(config_path)
    fmri_path = Path(config["fmri"]["path"]).resolve()
    if not fmri_path.exists():
        pytest.skip("BOLD Moments fmri path not available on this machine")

    # hcp_utils is required by RoiMasks.extract_rois(); skip if not installed
    try:
        import hcp_utils  # noqa: F401
    except Exception:
        pytest.skip("hcp_utils not available; skipping integration test")

    # Build dataset and load fmri
    dataset = build_dataset(config, split="train")
    fmri_data_by_subj = dataset.load_fmri_data()
    subject_id = dataset.subject_ids[0]
    fmri_data = fmri_data_by_subj[subject_id]

    # Verify noise ceiling file exists; else skip
    ncsnr_file = (
        fmri_path
        / f"{subject_id}/prepared_betas/{subject_id}_noiseceiling_task-train_n-3.pkl"
    )
    if not ncsnr_file.exists():
        pytest.skip("Noise ceiling file not available; skipping integration test")

    fmri_config = {
        "path": str(fmri_path),
        "sub_id": subject_id,
        "voxel_consistency_threshold": 0.3,
    }

    # Create masks using the helper
    masks = _load_and_create_masks(fmri_config, dataset.roi_masks, split="train")

    filtered = filter_by_voxel_consistency(
        fmri_data=fmri_data,
        masks=masks,
    )

    # Basic sanity: keys preserved; per-ROI voxel count not increased
    assert set(filtered.keys()) == set(fmri_data.keys())
    for roi in filtered.keys():
        assert _last_dim(filtered[roi]) <= _last_dim(fmri_data[roi])

    # Compute and print the ratio of voxels kept, per ROI
    for roi in filtered.keys():
        LOGGER.info(
            (
                f"Threshold: {fmri_config['voxel_consistency_threshold']}"
                f" Kept voxels for {roi}: "
                f"{_last_dim(filtered[roi]) / _last_dim(fmri_data[roi]):.2f}"
            )
        )


if __name__ == "__main__":
    coloredlogs.install(fmt="%(asctime)s %(name)s %(levelname)s %(message)s")
    tmp_path = Path("/tmp/test_voxel_consistency")
    tmp_path.mkdir(parents=True, exist_ok=True)
    test_filter_applies_threshold_per_voxel_and_roi(tmp_path=tmp_path, use_reps=False)
    test_filter_applies_threshold_per_voxel_and_roi(tmp_path=tmp_path, use_reps=True)
    test_uses_correct_noiseceiling_file_per_split(tmp_path=tmp_path)
    test_integration_on_bold_moments_if_available()
