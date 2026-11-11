"""Characterization tests locking the BOLD Moments dataset's observable behaviour.

The BOLD Moments numbers in the paper were produced by this pipeline, so any
refactor of the dataset layer must leave its behaviour untouched. These tests
pin the sample ordering - the one property every downstream score depends on,
because it is what pairs model features with fMRI rows.

They run against the real dataset and skip when it is not present.
"""

import hashlib
from pathlib import Path

import pytest
from benedict import benedict

from multitasking.datasets import build_dataset

CONFIG_PATH = "configs/test_configs/test_config_bold_moments_pipeline.yaml"

# Golden values captured from the dataset as shipped, before the NSD work began.
# The digest is over ",".join(sample_ids), so it pins membership *and* order.
GOLDEN = {
    "train": {
        "n": 998,
        "first": ["0001", "0002", "0003"],
        "last": ["0998", "0999", "1000"],
        "sha256": "5e2a5b475ab0210407fe153abdb197e56a86e4da1c516d4b6278f42db26df825",
    },
    "test": {
        "n": 96,
        "first": ["1002", "1003", "1004"],
        "last": ["1100", "1101", "1102"],
        "sha256": "05d8c0beb70ff007b54a51cb76390996bc83cc89923e722d4b80913f70370cbb",
    },
}


def _resolve(relative_path: str) -> Path:
    """Resolves a repo-relative path whether running from the root or tests/."""
    path = Path(relative_path)
    return path if path.exists() else Path("..") / relative_path


def _build(split: str):
    """Builds the BOLD Moments dataset for a split, skipping if data is absent."""
    config_path = _resolve(CONFIG_PATH)
    if not config_path.exists():
        pytest.skip("Test config not available")

    config = benedict.from_yaml(config_path)
    if not Path(config["fmri"]["path"]).exists():
        pytest.skip("BOLD Moments fmri path not available on this machine")
    if not Path(config["dataset"]["path"]).exists():
        pytest.skip("BOLD Moments stimuli not available on this machine")

    return build_dataset(config, split=split)


@pytest.mark.parametrize("split", ["train", "test"])
def test_sample_ids_match_golden_ordering(split: str):
    dataset = _build(split)
    expected = GOLDEN[split]

    sample_ids = dataset.sample_ids
    digest = hashlib.sha256(",".join(sample_ids).encode()).hexdigest()

    assert len(sample_ids) == expected["n"]
    assert sample_ids[:3] == expected["first"]
    assert sample_ids[-3:] == expected["last"]
    assert digest == expected["sha256"], (
        "BOLD Moments sample ordering changed. Every published score depends on "
        "this ordering, so this must not change without a deliberate decision."
    )


@pytest.mark.parametrize("split", ["train", "test"])
def test_len_matches_sample_ids(split: str):
    dataset = _build(split)
    assert len(dataset) == GOLDEN[split]["n"]
    assert len(dataset) == len(dataset.sample_ids)


def test_videos_follow_sample_ids():
    dataset = _build("test")
    video_ids = [path.stem for path in dataset.videos["test"]]
    assert video_ids == dataset.sample_ids


def test_sample_ids_are_sorted_and_unique():
    for split in ("train", "test"):
        sample_ids = _build(split).sample_ids
        assert sample_ids == sorted(sample_ids)
        assert len(set(sample_ids)) == len(sample_ids)


def test_fmri_rows_align_with_sample_ids():
    """The load path must return one fMRI row per sample, in sample_ids order."""
    dataset = _build("test")
    subject_id = dataset.subject_ids[0]

    betas, sample_ids = dataset._load_fmri_per_subject(
        subject_id, return_sample_ids=True
    )

    assert sample_ids == dataset.sample_ids
    assert betas.shape[0] == len(dataset.sample_ids)
