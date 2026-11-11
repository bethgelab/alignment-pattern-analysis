"""Tests for the cross-dataset ``test_split`` override.

The override exists so a model can be fitted on one dataset's train split and
scored on another's - NSD-core against NSD-synthetic. Three things have to hold
or the result is quietly wrong rather than loudly broken: it must apply to the
test split only, it must not mutate the caller's config, and it must refuse to
change anything the two splits have to agree on.
"""

import copy

import numpy as np
import pytest

from multitasking.datasets import (
    TEST_SPLIT_CONFIG_KEY,
    merge_test_split_override,
)
from multitasking.utils.file_handling import config_hash


def base_config():
    """Returns a minimal NSD-shaped config."""
    return {
        "dataset": {
            "name": "nsd",
            "path": "/data/NSD/stimuli",
            "resolution": 256,
            "surface_space": "fsaverage",
            "train_fraction": 0.8,
        },
        "fmri": {
            "path": "/data/NSD",
            "atlas_dir": "/data/NSD/atlas",
            "roi_beta_cache": "/data/NSD/cache",
            "sub_id": "all",
            "roi_names": ["V1", "MT"],
            "subsample_stride": 6,
        },
        "feature_extraction": {"model": "taskonomy/autoencoding"},
    }


def synthetic_override():
    """Returns a legal override pointing at NSD-synthetic."""
    return {
        "dataset": {
            "name": "nsd_synthetic",
            "path": "/data/NSD-synthetic/stimuli",
        },
        "fmri": {
            "path": "/data/NSD-synthetic/raw",
            "roi_beta_cache": "/data/NSD-synthetic/cache",
            "ncsnr_source": "nsd_synthetic",
            "nsd_core_path": "/data/NSD",
        },
    }


# --- merging ----------------------------------------------------------------


def test_merging_does_not_mutate_the_caller_config():
    """benchmark.py reads the config again after building the datasets."""
    config = base_config()
    before = copy.deepcopy(config)
    merge_test_split_override(config, synthetic_override())
    assert config == before


def test_overriding_a_shared_setting_is_refused():
    """Refuses to override a setting the two splits have to agree on.

    These are read from the train dataset or from the base config and applied
    to both splits, so letting the test split differ would de-synchronise them.
    """
    override = synthetic_override()
    override["fmri"]["roi_names"] = ["V1"]
    with pytest.raises(ValueError, match="cannot override"):
        merge_test_split_override(base_config(), override)


# --- dispatch ---------------------------------------------------------------


def test_build_dataset_applies_the_override_to_the_test_split_only(
    monkeypatch,
):
    import multitasking.datasets as datasets_module

    # Imported before the patches below: NSDSynthetic subclasses NSD, so
    # replacing NSD first would break the class statement at import time.
    import multitasking.datasets_nsd_synthetic  # noqa: F401

    seen = {}

    class FakeDataset:
        def __init__(self, name):
            self.name = name

    def fake_nsd(**kwargs):
        seen[kwargs["split"]] = ("nsd", kwargs)
        return FakeDataset("nsd")

    def fake_synthetic(**kwargs):
        seen[kwargs["split"]] = ("nsd_synthetic", kwargs)
        return FakeDataset("nsd_synthetic")

    monkeypatch.setattr(
        "multitasking.datasets_nsd.NSD", lambda **kw: fake_nsd(**kw)
    )
    monkeypatch.setattr(
        "multitasking.datasets_nsd_synthetic.NSDSynthetic",
        lambda **kw: fake_synthetic(**kw),
    )

    config = base_config()
    config[TEST_SPLIT_CONFIG_KEY] = synthetic_override()

    train = datasets_module.build_dataset(config, split="train")
    test = datasets_module.build_dataset(config, split="test")

    assert train.name == "nsd"
    assert test.name == "nsd_synthetic"
    assert seen["train"][1]["stimulus_path"] == "/data/NSD/stimuli"
    assert seen["test"][1]["stimulus_path"] == "/data/NSD-synthetic/stimuli"
    # The ROI set and atlas are shared, which is what keeps voxels aligned.
    assert seen["train"][1]["rois"] == seen["test"][1]["rois"]
    assert seen["train"][1]["atlas_dir"] == seen["test"][1]["atlas_dir"]
    assert (
        seen["train"][1]["subsample_stride"]
        == seen["test"][1]["subsample_stride"]
    )


# --- hashing ----------------------------------------------------------------


def test_adding_the_key_does_not_change_existing_config_hashes():
    """Existing NSD and BOLD Moments outputs and caches must stay valid."""
    config = base_config()
    keys = ["dataset", "feature_extraction", "fmri", TEST_SPLIT_CONFIG_KEY]
    assert config_hash(config, keys) == config_hash(
        config, ["dataset", "feature_extraction", "fmri"]
    )


# --- the comparability guard ------------------------------------------------


class FakeMasks:
    """Minimal stand-in for RoiMasks."""

    def __init__(self, roi_names, vertices):
        self.roi_names = roi_names
        self.rois = {
            name: np.asarray(vertices[name]) for name in roi_names
        }
        self.voxels_per_roi = {
            name: len(vertices[name]) for name in roi_names
        }
        self.rois_to_indices = {}
        start = 0
        for name in roi_names:
            self.rois_to_indices[name] = (start, start + len(vertices[name]))
            start += len(vertices[name])


class FakeSplit:
    """Minimal stand-in for a dataset, for the comparability guard."""

    def __init__(self, roi_names, vertices, resolution=(256, 256)):
        self.roi_masks = FakeMasks(roi_names, vertices)
        self.resolution = resolution


def test_guard_catches_reordered_rois():
    """Silent otherwise: the mask is a positional vector in ROI order."""
    from multitasking.benchmark import check_splits_are_comparable

    vertices = {"V1": [0, 1, 2], "MT": [7, 8]}
    with pytest.raises(ValueError, match="different order"):
        check_splits_are_comparable(
            FakeSplit(["V1", "MT"], vertices), FakeSplit(["MT", "V1"], vertices)
        )
