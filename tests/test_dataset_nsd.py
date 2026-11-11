"""Tests for the NSD dataset.

The synthetic fixture builds a miniature NSD: a few stimulus PNGs and a
per-ROI beta cache with a controlled repetition pattern. The real-data test
is skipped when the dataset is not on the machine.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from multitasking.CONSTANTS import NSD_ROIS
from multitasking.nsd.prepare_roi_betas import cache_path
from multitasking.nsd.trial_counts import NSD_SHARED1000_SIZE

NSD_ROOT = Path("/mnt/lustre/work/bethge/bkr857/datasets/NSD")
STIMULUS_ROOT = Path("/mnt/lustre/work/bethge/bkr578/data/NSD/stimuli")
BETA_CACHE = Path("/mnt/lustre/work/bethge/bkr578/data/NSD/cache")

REAL_DATA_AVAILABLE = (
    NSD_ROOT.exists() and STIMULUS_ROOT.exists() and BETA_CACHE.exists()
)

ROIS = ["V1", "MT"]
VERTS = {"V1": 4, "MT": 2}


@pytest.fixture
def fake_nsd(tmp_path, monkeypatch):
    """Builds a miniature NSD and patches the ROI masks and trial counts."""
    import multitasking.datasets_nsd as nsd_module
    from multitasking.fmri_data.roi_utils import RoiMasks

    n_vertices = sum(VERTS.values())

    class FakeRoiMasks(RoiMasks):
        surface_space = "fsaverage"

        def __init__(self, roi_names, roi_groups=None, **kwargs):
            super().__init__(roi_names, roi_groups)
            self.subsample_stride = kwargs.get("subsample_stride", 6)

        def extract_rois(self):
            self.rois = {}
            self.voxels_per_roi = {}
            self.rois_to_indices = {}
            start = 0
            for roi_name in self.roi_names:
                count = VERTS[roi_name]
                self.rois[roi_name] = np.arange(start, start + count)
                self.voxels_per_roi[roi_name] = count
                self.rois_to_indices[roi_name] = (start, start + count)
                start += count
            self.hcp_roi_mask = np.ones(n_vertices, dtype=np.int_)

    monkeypatch.setattr(
        nsd_module, "build_roi_masks", lambda names, groups=None, **kw: FakeRoiMasks(names, groups, **kw)
    )

    # Repetition counts: subjA saw every stimulus 3 times; subjB missed the
    # last 200, exactly as a subject with fewer sessions would.
    counts = {
        "subjA": np.full(NSD_SHARED1000_SIZE, 3, dtype=np.uint8),
        "subjB": np.full(NSD_SHARED1000_SIZE, 3, dtype=np.uint8),
    }
    counts["subjB"][800:] = 1
    monkeypatch.setattr(
        nsd_module, "load_shared1000_trial_counts", lambda *a, **k: counts
    )
    monkeypatch.setattr(nsd_module, "NSD_SUBJECTS", ["subjA", "subjB"])

    # Stimuli: tiny PNGs, one per shared-1000 stimulus.
    images_dir = tmp_path / "stimuli" / "images"
    images_dir.mkdir(parents=True)
    for stimulus_id in range(1, NSD_SHARED1000_SIZE + 1):
        value = stimulus_id % 256
        Image.new("RGB", (4, 4), (value, value, value)).save(
            images_dir / f"{stimulus_id:04d}.png"
        )

    # Beta cache: each row carries its own stimulus id, so misalignment shows.
    cache_root = tmp_path / "cache"
    for subject_id, subject_counts in counts.items():
        betas = np.tile(
            np.arange(1, NSD_SHARED1000_SIZE + 1, dtype=np.float32)[:, None],
            (1, n_vertices),
        )
        target = cache_path(cache_root, subject_id, ROIS, "fsaverage", 6)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            betas=betas,
            trial_counts=subject_counts,
            stimulus_ids=np.arange(1, NSD_SHARED1000_SIZE + 1, dtype=np.uint16),
            roi_names=np.asarray(ROIS),
            roi_vertex_counts=np.asarray([VERTS[roi] for roi in ROIS]),
        )

    return {
        "stimulus_path": tmp_path / "stimuli",
        "fmri_path": tmp_path / "fmri",
        "roi_beta_cache": cache_root,
        "rois": ROIS,
    }


def _build(fake_nsd, split="test", rois_override=None, **kwargs):
    from multitasking.datasets_nsd import NSD

    return NSD(
        stimulus_path=fake_nsd["stimulus_path"],
        fmri_path=fake_nsd["fmri_path"],
        rois=rois_override or fake_nsd["rois"],
        split=split,
        roi_beta_cache=fake_nsd["roi_beta_cache"],
        **kwargs,
    )


def _rewrite_cache(dataset, subject_id, **changes):
    target = dataset._cache_path(subject_id)
    with np.load(target) as cached:
        contents = dict(cached)
    for key, change in changes.items():
        contents[key] = change(contents[key])
    np.savez_compressed(target, **contents)


# --- the sample-ordering contract -------------------------------------------


def test_sample_ids_do_not_depend_on_which_subjects_are_scored(fake_nsd):
    """Scoring fewer subjects must not change the stimulus set.

    Otherwise a pairwise inter-subject run and the benchmark run it is
    compared against would use different stimuli.
    """
    both = _build(fake_nsd, subject_ids=["subjA", "subjB"])
    one = _build(fake_nsd, subject_ids=["subjA"])
    assert one.sample_ids == both.sample_ids


def test_fmri_rows_follow_sample_ids(fake_nsd):
    """Each cached row carries its own stimulus id, so misalignment shows up."""
    dataset = _build(fake_nsd)
    fmri = dataset.load_fmri_data()
    expected = np.array([int(sample_id) for sample_id in dataset.sample_ids])
    for subject_data in fmri.values():
        for roi_data in subject_data.values():
            np.testing.assert_array_equal(roi_data[:, 0], expected)


def test_stimuli_are_yielded_in_sample_id_order(fake_nsd):
    dataset = _build(fake_nsd)
    seen = [
        sample_id
        for _, ids in dataset.load_videos(batch_size=7)
        for sample_id in ids
    ]
    assert seen == dataset.sample_ids


# --- the beta cache ---------------------------------------------------------


def test_mismatched_trial_counts_raise(fake_nsd, monkeypatch):
    """The derived stimulus set is re-checked against the loaded betas."""
    dataset = _build(fake_nsd)
    tampered = {
        "subjA": np.full(NSD_SHARED1000_SIZE, 3, dtype=np.uint8),
        "subjB": np.full(NSD_SHARED1000_SIZE, 3, dtype=np.uint8),
    }
    tampered["subjA"][:100] = 1
    monkeypatch.setattr(dataset, "_trial_counts", tampered)

    with pytest.raises(ValueError, match="disagree with the counts derived"):
        dataset.load_fmri_data()


def test_cache_with_extra_rois_serves_a_subset(fake_nsd):
    """A cache built once with the full ROI list must serve any subset."""
    full = _build(fake_nsd)
    subset = _build(fake_nsd, rois_override=["MT"])

    full_fmri = full.load_fmri_data()
    subset_fmri = subset.load_fmri_data()

    assert set(subset_fmri["subjA"]) == {"MT"}
    np.testing.assert_array_equal(
        subset_fmri["subjA"]["MT"], full_fmri["subjA"]["MT"]
    )


def test_cache_with_wrong_vertex_count_raises(fake_nsd):
    """A cache built with a different atlas or stride must not be read."""
    dataset = _build(fake_nsd)
    _rewrite_cache(
        dataset,
        "subjA",
        roi_vertex_counts=lambda _: np.asarray([VERTS["V1"] + 1, VERTS["MT"] - 1]),
    )
    with pytest.raises(ValueError, match="different atlas or stride"):
        dataset.load_fmri_data()


def test_missing_cells_are_filled_but_a_missing_vertex_raises(fake_nsd):
    """A missing cell is filled with zero, but a missing vertex raises.

    Filling a vertex with no data at all would create a constant feature.
    """
    dataset = _build(fake_nsd)

    def one_missing_cell(betas):
        betas[5, 1] = np.nan
        return betas

    _rewrite_cache(dataset, "subjA", betas=one_missing_cell)
    for roi_data in dataset.load_fmri_data()["subjA"].values():
        assert not np.isnan(roi_data).any()

    def one_missing_vertex(betas):
        betas[:, 1] = np.nan
        return betas

    _rewrite_cache(dataset, "subjA", betas=one_missing_vertex)
    with pytest.raises(ValueError, match="no valid value for any stimulus"):
        dataset.load_fmri_data()


# --- real data --------------------------------------------------------------


@pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="NSD data not available")
def test_real_nsd_fmri_alignment():
    from multitasking.datasets_nsd import NSD

    dataset = NSD(
        stimulus_path=STIMULUS_ROOT,
        fmri_path=NSD_ROOT,
        rois=NSD_ROIS,
        split="test",
        subject_ids=["subj01", "subj04"],
        roi_beta_cache=BETA_CACHE,
    )
    fmri = dataset.load_fmri_data()

    for subject_id, subject_data in fmri.items():
        for roi_name, roi_data in subject_data.items():
            assert roi_data.shape[0] == len(dataset), (subject_id, roi_name)
            assert not np.isnan(roi_data).any()
            assert roi_data.shape[1] == dataset.roi_masks.voxels_per_roi[roi_name]
