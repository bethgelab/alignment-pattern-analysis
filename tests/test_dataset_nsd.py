"""Tests for the NSD dataset.

The synthetic fixtures build a miniature NSD: a few stimulus PNGs, a per-ROI
beta cache, and an NCSNR file, with a controlled repetition pattern. The
real-data tests are skipped when the dataset is not on the machine.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from multitasking.CONSTANTS import NSD_ROIS, NSD_SUBJECTS
from multitasking.datasets_base import FmriStimulusDataset
from multitasking.nsd.prepare_roi_betas import cache_path
from multitasking.nsd.trial_counts import NSD_SHARED1000_SIZE

NSD_ROOT = Path("/mnt/lustre/work/bethge/bkr857/datasets/NSD")
STIMULUS_ROOT = Path("/mnt/lustre/work/bethge/bkr578/data/NSD/stimuli")
BETA_CACHE = Path("/mnt/lustre/work/bethge/bkr578/data/NSD/cache")

REAL_DATA_AVAILABLE = (
    NSD_ROOT.exists() and STIMULUS_ROOT.exists() and BETA_CACHE.exists()
)
requires_real_data = pytest.mark.skipif(
    not REAL_DATA_AVAILABLE, reason="NSD data not available on this machine"
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

    # Stimuli: tiny PNGs whose single pixel encodes the stimulus id.
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
        )

    # NCSNR, in full-surface vertex space.
    for subject_id in counts:
        ncsnr_dir = (
            tmp_path
            / "fmri"
            / "nsddata_betas"
            / "ppdata"
            / subject_id
            / "fsaverage"
            / "betas_fithrf_GLMdenoise_RR"
            / "prepared_betas"
        )
        ncsnr_dir.mkdir(parents=True, exist_ok=True)
        half = n_vertices // 2
        np.savez(
            ncsnr_dir / f"{subject_id}_ncsnr.npz",
            ncsnr_lh=np.linspace(0, 1, half),
            ncsnr_rh=np.linspace(0, 1, n_vertices - half),
        )

    return {
        "stimulus_path": tmp_path / "stimuli",
        "fmri_path": tmp_path / "fmri",
        "roi_beta_cache": cache_root,
        "rois": ROIS,
        "subjects": ["subjA", "subjB"],
    }


def _build(fake_nsd, split="test", **kwargs):
    from multitasking.datasets_nsd import NSD

    return NSD(
        stimulus_path=fake_nsd["stimulus_path"],
        fmri_path=fake_nsd["fmri_path"],
        rois=fake_nsd["rois"],
        split=split,
        roi_beta_cache=fake_nsd["roi_beta_cache"],
        **kwargs,
    )


# --- the contract -----------------------------------------------------------


def test_nsd_satisfies_the_dataset_contract(fake_nsd):
    dataset = _build(fake_nsd)
    assert isinstance(dataset, FmriStimulusDataset)
    assert len(dataset) == len(dataset.sample_ids)


def test_sample_ids_are_zero_padded_and_sorted(fake_nsd):
    dataset = _build(fake_nsd)
    assert all(len(sample_id) == 4 for sample_id in dataset.sample_ids)
    assert dataset.sample_ids == sorted(dataset.sample_ids)


# --- the key invariant ------------------------------------------------------


def test_sample_ids_do_not_depend_on_which_subjects_are_scored(fake_nsd):
    """Scoring fewer subjects must not change the stimulus set.

    Otherwise a pairwise inter-subject run and the benchmark run it is
    compared against would use different stimuli.
    """
    both = _build(fake_nsd, subject_ids=["subjA", "subjB"])
    one = _build(fake_nsd, subject_ids=["subjA"])
    assert one.sample_ids == both.sample_ids


def test_stimulus_subject_ids_does_change_the_stimulus_set(fake_nsd):
    """The set follows stimulus_subject_ids, which is the knob that controls it."""
    with_b = _build(fake_nsd, stimulus_subject_ids=["subjA", "subjB"])
    without_b = _build(
        fake_nsd, subject_ids=["subjA"], stimulus_subject_ids=["subjA"]
    )
    assert len(without_b.sample_ids) > len(with_b.sample_ids)


def test_scoring_a_subject_outside_the_stimulus_set_raises(fake_nsd):
    with pytest.raises(ValueError, match="stimulus_subject_ids"):
        _build(
            fake_nsd, subject_ids=["subjB"], stimulus_subject_ids=["subjA"]
        )


def test_splits_are_disjoint_and_cover_the_usable_stimuli(fake_nsd):
    train = _build(fake_nsd, split="train")
    test = _build(fake_nsd, split="test")
    assert set(train.sample_ids).isdisjoint(test.sample_ids)
    # subjB missed the last 200 stimuli, so 800 remain usable.
    assert len(train) + len(test) == 800


# --- fMRI alignment ---------------------------------------------------------


def test_fmri_rows_follow_sample_ids(fake_nsd):
    """Each cached row carries its own stimulus id, so misalignment shows up."""
    dataset = _build(fake_nsd)
    fmri = dataset.load_fmri_data()
    expected = np.array([int(sample_id) for sample_id in dataset.sample_ids])
    for subject_data in fmri.values():
        for roi_data in subject_data.values():
            np.testing.assert_array_equal(roi_data[:, 0], expected)


def test_fmri_is_split_per_roi_with_correct_widths(fake_nsd):
    dataset = _build(fake_nsd)
    fmri = dataset.load_fmri_data()
    for subject_data in fmri.values():
        assert set(subject_data) == set(ROIS)
        for roi_name, roi_data in subject_data.items():
            assert roi_data.shape == (len(dataset), VERTS[roi_name])


def test_mismatched_trial_counts_raise(fake_nsd, monkeypatch):
    """The derived stimulus set is re-checked against the loaded betas."""
    import multitasking.datasets_nsd as nsd_module

    dataset = _build(fake_nsd)
    tampered = {
        "subjA": np.full(NSD_SHARED1000_SIZE, 3, dtype=np.uint8),
        "subjB": np.full(NSD_SHARED1000_SIZE, 3, dtype=np.uint8),
    }
    tampered["subjA"][:100] = 1
    monkeypatch.setattr(dataset, "_trial_counts", tampered)

    with pytest.raises(ValueError, match="disagree with the counts derived"):
        dataset.load_fmri_data()
    assert nsd_module is not None


def test_cache_built_for_other_rois_raises(fake_nsd):
    """A cache at the expected path whose stored ROI list disagrees."""
    dataset = _build(fake_nsd)
    target = dataset._cache_path("subjA")
    with np.load(target) as cached:
        contents = dict(cached)
    contents["roi_names"] = np.asarray(["V1", "SomethingElse"])
    np.savez_compressed(target, **contents)

    with pytest.raises(ValueError, match="built for ROIs"):
        dataset.load_fmri_data()


def test_missing_cache_raises_with_guidance(fake_nsd, tmp_path):
    from multitasking.datasets_nsd import NSD

    dataset = NSD(
        stimulus_path=fake_nsd["stimulus_path"],
        fmri_path=fake_nsd["fmri_path"],
        rois=fake_nsd["rois"],
        split="test",
        roi_beta_cache=tmp_path / "absent",
    )
    with pytest.raises(FileNotFoundError, match="prepare_roi_betas"):
        dataset.load_fmri_data()


# --- stimuli ----------------------------------------------------------------


def test_stimuli_are_served_with_a_singleton_time_axis(fake_nsd):
    dataset = _build(fake_nsd)
    batch, ids = next(iter(dataset.load_videos(batch_size=3)))
    assert batch.shape[0] == 3
    assert batch.shape[1] == 3  # channels
    assert batch.shape[2] == 1  # T
    assert batch.dtype.__str__() == "torch.uint8"
    assert ids == dataset.sample_ids[:3]


def test_all_stimulus_ids_are_yielded_in_order(fake_nsd):
    dataset = _build(fake_nsd)
    seen = [
        sample_id
        for _, ids in dataset.load_videos(batch_size=7)
        for sample_id in ids
    ]
    assert seen == dataset.sample_ids


def test_resolution_is_applied(fake_nsd):
    dataset = _build(fake_nsd, resolution=8)
    batch, _ = next(iter(dataset.load_videos(batch_size=1)))
    assert batch.shape[-2:] == (8, 8)


def test_missing_stimulus_raises_with_guidance(fake_nsd):
    dataset = _build(fake_nsd)
    target = dataset.stimulus_path / "images" / f"{dataset.sample_ids[0]}.png"
    target.unlink()
    with pytest.raises(FileNotFoundError, match="prepare_stimuli"):
        next(iter(dataset.load_videos(batch_size=1)))


def test_voxel_consistency_matches_roi_widths(fake_nsd):
    dataset = _build(fake_nsd)
    ncsnr = dataset.load_voxel_consistency("subjA")
    assert set(ncsnr) == set(ROIS)
    for roi_name, values in ncsnr.items():
        assert values.shape == (VERTS[roi_name],)


# --- real data --------------------------------------------------------------


@requires_real_data
def test_real_nsd_default_configuration():
    from multitasking.datasets_nsd import NSD

    common = dict(
        stimulus_path=STIMULUS_ROOT,
        fmri_path=NSD_ROOT,
        rois=NSD_ROIS,
        roi_beta_cache=BETA_CACHE,
        resolution=256,
    )
    train = NSD(split="train", **common)
    test = NSD(split="test", **common)

    assert len(train) == 618
    assert len(test) == 148
    assert set(train.sample_ids).isdisjoint(test.sample_ids)
    assert train.subject_ids == list(NSD_SUBJECTS)


@requires_real_data
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
