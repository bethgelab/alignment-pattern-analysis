import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from benedict import benedict

from multitasking.datasets import BoldMoments, build_dataset


@pytest.fixture
def mock_boldmoments_dataset(tmp_path):
    # Create mock video directory and files
    video_dir = tmp_path / "mp4_h264"
    video_dir.mkdir(parents=True)
    # Create 10 mock video files
    for i in range(10):
        (video_dir / f"{i:04d}.mp4").touch()
    # Create mock fmri directory with subject folders
    fmri_dir = tmp_path / "fmri"
    fmri_dir.mkdir()
    for sub in ["sub-01", "sub-02"]:
        (fmri_dir / sub).mkdir()
    # Return paths
    return {
        "video_path": video_dir.parent,
        "fmri_path": fmri_dir,
        "rois": ["V1", "V2"],
        "split": "train",
        "subject_ids": "all",
        "aggregate_fn": "avg",
        "resolution": 256,
        "frame_rate": 4,
    }


def test_boldmoments_init_and_len(mock_boldmoments_dataset):
    # Should not raise
    ds = BoldMoments(
        video_path=mock_boldmoments_dataset["video_path"],
        fmri_path=mock_boldmoments_dataset["fmri_path"],
        rois=mock_boldmoments_dataset["rois"],
        split=mock_boldmoments_dataset["split"],
        subject_ids=mock_boldmoments_dataset["subject_ids"],
        aggregate_fn=mock_boldmoments_dataset["aggregate_fn"],
        resolution=mock_boldmoments_dataset["resolution"],
        frame_rate=mock_boldmoments_dataset["frame_rate"],
    )
    # There should be 10 videos in the train split (minus any invalids, but none
    # in this mock)
    assert len(ds) == 10
    # The split should be set correctly
    assert ds.split == "train"
    # The subject_ids should be detected from fmri_path
    assert sorted(ds.subject_ids) == ["sub-01", "sub-02"]
    # The rois should be set
    assert ds.rois == ["V1", "V2"]
    # The resolution and frame_rate should be set
    assert ds.resolution == (256, 256)
    assert ds.frame_rate == 4


def test_boldmoments_getitem_raises(mock_boldmoments_dataset):
    ds = BoldMoments(
        video_path=mock_boldmoments_dataset["video_path"],
        fmri_path=mock_boldmoments_dataset["fmri_path"],
        rois=mock_boldmoments_dataset["rois"],
        split=mock_boldmoments_dataset["split"],
        subject_ids=mock_boldmoments_dataset["subject_ids"],
        aggregate_fn=mock_boldmoments_dataset["aggregate_fn"],
        resolution=mock_boldmoments_dataset["resolution"],
        frame_rate=mock_boldmoments_dataset["frame_rate"],
    )
    with pytest.raises(NotImplementedError):
        _ = ds[0]


def test_boldmoments_invalid_aggregate_fn(mock_boldmoments_dataset):
    with pytest.raises(NotImplementedError):
        BoldMoments(
            video_path=mock_boldmoments_dataset["video_path"],
            fmri_path=mock_boldmoments_dataset["fmri_path"],
            rois=mock_boldmoments_dataset["rois"],
            split=mock_boldmoments_dataset["split"],
            subject_ids=mock_boldmoments_dataset["subject_ids"],
            aggregate_fn="median",  # Not implemented
            resolution=mock_boldmoments_dataset["resolution"],
            frame_rate=mock_boldmoments_dataset["frame_rate"],
        )


# Test that fmri data and videos are returned in the same order for the same
# sample ids
def test_boldmoments_video_and_fmri_order_consistency(tmp_path):
    # Setup a mock dataset with known sample ids
    video_path = tmp_path / "stimulus_set"
    fmri_path = tmp_path / "fmri"
    video_path.mkdir(parents=True)
    (video_path / "mp4_h264").mkdir(parents=True)
    (fmri_path / "sub-01").mkdir(parents=True)
    (fmri_path / "sub-02").mkdir(parents=True)
    (fmri_path / "sub-01" / "prepared_betas").mkdir(parents=True)
    (fmri_path / "sub-02" / "prepared_betas").mkdir(parents=True)
    rois = ["V1", "V2"]
    split = "train"
    subject_ids = ["sub-01", "sub-02"]
    aggregate_fn = "avg"
    resolution = 256
    frame_rate = 4

    # Create 5 mock video files
    sample_ids = [f"{i:04d}" for i in range(1, 6)]
    for sid in sample_ids:
        (video_path / "mp4_h264" / f"{sid}.mp4").touch()

    # Create mock fMRI data
    for subject_id in subject_ids:
        # Create data
        mock_data = np.random.rand(5, 10, 91282)
        mock_sample_ids = [f"vid{i:04d}" for i in range(1, 6)]
        # Create pkl file and dump data
        tmp_f = (
            fmri_path
            / subject_id
            / "prepared_betas"
            / f"{subject_id}_organized_betas_task-{split}_normalized.pkl"
        )
        with open(tmp_f, "wb") as f:
            pickle.dump((mock_data, mock_sample_ids), f)

    # Create a simple mock RoiMasks class to avoid actual ROI extraction
    class MockRoiMasks:
        def __init__(self, rois):
            self.rois = {roi: np.arange(100) for roi in rois}

        def extract_rois(self):
            pass

    # Monkey patch the RoiMasks import
    import multitasking.datasets

    original_roi_masks = multitasking.datasets.RoiMasks
    multitasking.datasets.RoiMasks = MockRoiMasks

    try:
        ds = BoldMoments(
            video_path=video_path,
            fmri_path=fmri_path,
            rois=rois,
            split=split,
            subject_ids=subject_ids,
            aggregate_fn=aggregate_fn,
            resolution=resolution,
            frame_rate=frame_rate,
        )

        # Mock the _load_video method to return a dummy tensor
        def mock_load_video(video_path):
            return torch.randn(3, 12, 40, 40)

        # Store the original method and replace it
        original_load_video = ds._load_video
        ds._load_video = mock_load_video

        try:
            # Get fmri data and videos
            fmri_retrieved, fmri_sample_ids_retrieved = \
                ds._load_fmri_per_subject(
                "sub-01", return_sample_ids=True
            )

            # Get video data - collect all videos with a large batch size to get
            # all at once
            videos, video_sample_ids = next(
                ds.load_videos(batch_size=10)
            )  # Use batch_size >= number of videos

            # Check that the sample IDs are in the expected order
            expected_sample_ids = [f"{i:04d}" for i in range(1, 6)]
            assert fmri_sample_ids_retrieved == expected_sample_ids, (
                f"FMRI sample IDs {fmri_sample_ids_retrieved}"
                f"do not match expected {expected_sample_ids}"
            )

            # Check that the video sample IDs match the fMRI sample IDs
            assert video_sample_ids == fmri_sample_ids_retrieved, (
                f"Video sample IDs {video_sample_ids} do not match fMRI sample "
                f"IDs {fmri_sample_ids_retrieved}"
            )

            # Check that the dataset length is correct
            assert len(ds) == 5, f"Dataset length {len(ds)} does not match "
            "expected 5"

        finally:
            # Restore the original _load_video method
            ds._load_video = original_load_video

    finally:
        # Restore the original RoiMasks class
        multitasking.datasets.RoiMasks = original_roi_masks



def test_subject_id_parsing():
    # Resolve path whether running from root or tests directory
    config_path_str = "configs/test_configs/test_config_bold_moments_pipeline.yaml"
    config_path = Path(config_path_str)
    if not config_path.exists():
        config_path = Path("..") / config_path_str
    if not config_path.exists():
        raise FileNotFoundError(f"Test config not available at {config_path_str}")

    config = benedict.from_yaml(config_path)
    assert config["fmri"]["sub_id"].startswith("sub-"), \
          "Expected single sub_id in test config"
    sub_id = config["fmri"]["sub_id"]
    fmri_path = Path(config["fmri"]["path"]).resolve()
    if not fmri_path.exists():
        raise FileNotFoundError(f"BOLD Moments fmri path not available at {fmri_path}")

    # Build dataset and load fmri
    dataset = build_dataset(config, split="train")
    subject_id_dataset = dataset.subject_ids[0]
    assert subject_id_dataset == sub_id, (
        f"Expected {sub_id} but got {subject_id_dataset}, all "
        f"subject ids: {dataset.subject_ids}; failed to parse single subject id "
        "in dataset creation"
    )


def _build_ordering_dataset(tmp_path, monkeypatch, stored_ids, stored_data):
    """Builds a BoldMoments dataset over 5 mock train stimuli.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.
        stored_ids: Sample IDs to store in the mock fMRI pickle (e.g. "vid0001").
        stored_data: fMRI betas to store, aligned with stored_ids.

    Returns:
        The constructed BoldMoments dataset.
    """
    import multitasking.datasets

    video_path = tmp_path / "stimulus_set"
    fmri_path = tmp_path / "fmri"
    (video_path / "mp4_h264").mkdir(parents=True)
    (fmri_path / "sub-01" / "prepared_betas").mkdir(parents=True)

    for sample_id in [f"{i:04d}" for i in range(1, 6)]:
        (video_path / "mp4_h264" / f"{sample_id}.mp4").touch()

    betas_file = (
        fmri_path
        / "sub-01"
        / "prepared_betas"
        / "sub-01_organized_betas_task-train_normalized.pkl"
    )
    with open(betas_file, "wb") as f:
        pickle.dump((stored_data, stored_ids), f)

    class MockRoiMasks:
        def __init__(self, rois):
            self.rois = {roi: np.arange(20) for roi in rois}
            self.roi_names = list(rois)

        def extract_rois(self):
            pass

    monkeypatch.setattr(multitasking.datasets, "RoiMasks", MockRoiMasks)

    return BoldMoments(
        video_path=video_path,
        fmri_path=fmri_path,
        rois=["V1"],
        split="train",
        subject_ids=["sub-01"],
    )


def test_fmri_rows_are_reordered_to_video_order(tmp_path, monkeypatch):
    # The pickle stores the samples in a shuffled order; each row carries its own
    # sample ID as its value, so misalignment is detectable.
    sample_ids = [f"{i:04d}" for i in range(1, 6)]
    shuffled = [3, 0, 4, 1, 2]
    stored_data = np.stack(
        [np.full((10, 20), float(sample_ids[i])) for i in shuffled]
    )
    stored_ids = [f"vid{sample_ids[i]}" for i in shuffled]

    ds = _build_ordering_dataset(tmp_path, monkeypatch, stored_ids, stored_data)

    fmri_data, fmri_sample_ids = ds._load_fmri_per_subject(
        "sub-01", return_sample_ids=True
    )

    assert ds.sample_ids == sample_ids
    assert fmri_sample_ids == sample_ids
    # Row i must still carry the signature of sample_ids[i] after reordering.
    np.testing.assert_allclose(
        fmri_data[:, 0], [float(sample_id) for sample_id in sample_ids]
    )


def test_missing_fmri_sample_raises(tmp_path, monkeypatch):
    # The pickle is missing stimulus 0003 entirely.
    present_ids = ["0001", "0002", "0004", "0005"]
    stored_data = np.stack(
        [np.full((10, 20), float(sample_id)) for sample_id in present_ids]
    )
    stored_ids = [f"vid{sample_id}" for sample_id in present_ids]

    ds = _build_ordering_dataset(tmp_path, monkeypatch, stored_ids, stored_data)

    with pytest.raises(ValueError, match="missing"):
        ds._load_fmri_per_subject("sub-01")


def test_split_is_derived_from_sample_id(tmp_path, monkeypatch):
    # Stimulus 1001 belongs to the test split even though only 6 videos exist.
    import multitasking.datasets

    video_path = tmp_path / "stimulus_set"
    fmri_path = tmp_path / "fmri"
    (video_path / "mp4_h264").mkdir(parents=True)
    (fmri_path / "sub-01").mkdir(parents=True)

    for sample_id in ["0001", "0002", "0003", "1001", "1002", "1003"]:
        (video_path / "mp4_h264" / f"{sample_id}.mp4").touch()

    class MockRoiMasks:
        def __init__(self, rois):
            self.rois = {roi: np.arange(20) for roi in rois}
            self.roi_names = list(rois)

        def extract_rois(self):
            pass

    monkeypatch.setattr(multitasking.datasets, "RoiMasks", MockRoiMasks)

    ds = BoldMoments(
        video_path=video_path,
        fmri_path=fmri_path,
        rois=["V1"],
        split="test",
        subject_ids=["sub-01"],
    )

    assert ds.all_sample_ids_by_split["train"] == ["0001", "0002", "0003"]
    assert ds.all_sample_ids_by_split["test"] == ["1001", "1002", "1003"]
    # "1001" is on the invalid list for the test split.
    assert ds.sample_ids == ["1002", "1003"]
