"""Tests for the NSD-synthetic dataset.

Built like ``tests/test_dataset_nsd.py``: a miniature dataset on disk - stimulus
PNGs, a per-ROI beta cache, an NCSNR file - with rows that carry their own
stimulus ID so that any misalignment shows up as a wrong number rather than as a
plausible one. The real-data tests skip when the dataset is not on the machine.

The repetition counts come from the checked-in design metadata rather than from
a fixture, since they are a property of the published experiment.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from multitasking.CONSTANTS import NSD_SUBJECTS
from multitasking.datasets_base import FmriStimulusDataset
from multitasking.nsd.prepare_roi_betas import NSD_STIMULUS_SET, cache_path
from multitasking.nsd_synthetic.design import (
    NSD_SYNTHETIC_N_ACHROMATIC,
    NSD_SYNTHETIC_STIMULUS_SET,
    load_trial_counts,
)
from multitasking.nsd_synthetic.prepare_betas import (
    NSD_SYNTHETIC_BETAS_DIRNAME,
    NSD_SYNTHETIC_CACHE_DIRNAME,
)
from multitasking.nsd_synthetic.prepare_stimuli import PROVENANCE_FILENAME

SYNTHETIC_ROOT = Path("/mnt/lustre/work/bethge/bkr578/data/NSD-synthetic")
STIMULUS_ROOT = SYNTHETIC_ROOT / "stimuli"
BETA_CACHE = SYNTHETIC_ROOT / "cache"
NSD_CORE_ROOT = Path("/mnt/lustre/work/bethge/bkr857/datasets/NSD")

REAL_DATA_AVAILABLE = (
    STIMULUS_ROOT.exists()
    and BETA_CACHE.exists()
    and (SYNTHETIC_ROOT / "raw").exists()
)
requires_real_data = pytest.mark.skipif(
    not REAL_DATA_AVAILABLE,
    reason="NSD-synthetic data not available on this machine",
)

ROIS = ["V1", "MT"]
VERTS = {"V1": 4, "MT": 2}
SUBJECTS = ["subjA", "subjB"]


@pytest.fixture
def fake_synthetic(tmp_path, monkeypatch):
    """Builds a miniature NSD-synthetic and patches the ROI masks."""
    import multitasking.datasets_nsd as nsd_module
    from multitasking.fmri_data.roi_utils import RoiMasks

    n_vertices = sum(VERTS.values())
    n_stimuli = NSD_SYNTHETIC_N_ACHROMATIC
    design_counts = load_trial_counts()[:n_stimuli]

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

    # NSDSynthetic reaches these through super().__init__, so patching the
    # parent module covers both classes with one fixture.
    monkeypatch.setattr(
        nsd_module,
        "build_roi_masks",
        lambda names, groups=None, **kw: FakeRoiMasks(names, groups, **kw),
    )
    monkeypatch.setattr(nsd_module, "NSD_SUBJECTS", SUBJECTS)

    # Stimuli, under the syn namespace, plus the provenance tag.
    stimulus_path = tmp_path / "stimuli"
    images_dir = stimulus_path / "images"
    images_dir.mkdir(parents=True)
    for stimulus_id in range(1, n_stimuli + 1):
        value = stimulus_id % 256
        Image.new("RGB", (4, 4), (value, value, value)).save(
            images_dir / f"syn{stimulus_id:04d}.png"
        )
    (stimulus_path / PROVENANCE_FILENAME).write_text(
        json.dumps({"stimulus_set": NSD_SYNTHETIC_STIMULUS_SET})
    )

    cache_root = tmp_path / "cache"
    for subject_id in SUBJECTS:
        write_cache(cache_root, subject_id, design_counts, n_vertices)

    fmri_path = tmp_path / "fmri"
    for subject_id in SUBJECTS:
        write_ncsnr(
            fmri_path, subject_id, NSD_SYNTHETIC_BETAS_DIRNAME, n_vertices
        )

    # An NSD-core tree, so the nsd_core NCSNR source has something to read.
    core_path = tmp_path / "nsd_core"
    for subject_id in SUBJECTS:
        write_ncsnr(
            core_path,
            subject_id,
            "betas_fithrf_GLMdenoise_RR",
            n_vertices,
            scale=0.5,
        )

    return {
        "stimulus_path": stimulus_path,
        "fmri_path": fmri_path,
        "nsd_core_fmri_path": core_path,
        "roi_beta_cache": cache_root,
        "rois": ROIS,
        "n_stimuli": n_stimuli,
        "design_counts": design_counts,
        "n_vertices": n_vertices,
    }


def write_cache(
    cache_root,
    subject_id,
    trial_counts,
    n_vertices,
    *,
    rois=None,
    stimulus_set=NSD_SYNTHETIC_STIMULUS_SET,
    dirname=NSD_SYNTHETIC_CACHE_DIRNAME,
    max_reps=-1,
):
    """Writes one per-ROI beta cache whose rows carry their stimulus ID."""
    rois = rois or ROIS
    n_stimuli = len(trial_counts)
    width = sum(VERTS[roi] for roi in rois)
    betas = np.tile(
        np.arange(1, n_stimuli + 1, dtype=np.float32)[:, None], (1, width)
    )
    target = cache_path(
        cache_root, subject_id, rois, "fsaverage", 6, dirname=dirname
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        betas=betas,
        trial_counts=np.asarray(trial_counts, dtype=np.uint8),
        stimulus_ids=np.arange(1, n_stimuli + 1, dtype=np.uint16),
        roi_names=np.asarray(rois),
        roi_vertex_counts=np.asarray([VERTS[roi] for roi in rois]),
        stimulus_set=np.asarray(stimulus_set),
        max_reps=np.asarray(max_reps),
    )
    del n_vertices
    return target


def write_ncsnr(fmri_path, subject_id, betas_dirname, n_vertices, scale=1.0):
    """Writes one whole-surface NCSNR file."""
    ncsnr_dir = (
        fmri_path
        / "nsddata_betas"
        / "ppdata"
        / subject_id
        / "fsaverage"
        / betas_dirname
        / "prepared_betas"
    )
    ncsnr_dir.mkdir(parents=True, exist_ok=True)
    half = n_vertices // 2
    np.savez(
        ncsnr_dir / f"{subject_id}_ncsnr.npz",
        ncsnr_lh=np.linspace(0, 1, half) * scale,
        ncsnr_rh=np.linspace(0, 1, n_vertices - half) * scale,
    )


def _build(fake_synthetic, split="test", rois_override=None, **kwargs):
    from multitasking.datasets_nsd_synthetic import NSDSynthetic

    return NSDSynthetic(
        stimulus_path=fake_synthetic["stimulus_path"],
        fmri_path=fake_synthetic["fmri_path"],
        rois=rois_override or fake_synthetic["rois"],
        split=split,
        roi_beta_cache=fake_synthetic["roi_beta_cache"],
        **kwargs,
    )


# --- the contract -----------------------------------------------------------


def test_satisfies_the_dataset_contract(fake_synthetic):
    dataset = _build(fake_synthetic)
    assert isinstance(dataset, FmriStimulusDataset)
    assert len(dataset) == fake_synthetic["n_stimuli"]
    assert len(dataset.sample_ids) == len(dataset)
    assert dataset.subject_ids == SUBJECTS


def test_sample_ids_are_syn_namespaced_and_ordered(fake_synthetic):
    """A shared namespace with NSD-core would be silently wrong, not loud."""
    dataset = _build(fake_synthetic)
    assert dataset.sample_ids[0] == "syn0001"
    assert dataset.sample_ids[-1] == f"syn{fake_synthetic['n_stimuli']:04d}"
    assert dataset.sample_ids == sorted(dataset.sample_ids)
    assert all(sample_id.startswith("syn") for sample_id in dataset.sample_ids)


def test_sample_ids_do_not_depend_on_which_subjects_are_scored(fake_synthetic):
    both = _build(fake_synthetic).sample_ids
    one = _build(fake_synthetic, subject_ids=["subjB"]).sample_ids
    assert both == one


def test_there_is_no_train_split(fake_synthetic):
    """NSD-synthetic is one session; it is scored, not fitted on."""
    with pytest.raises(ValueError, match="no 'train' split"):
        _build(fake_synthetic, split="train")


# --- stimuli ----------------------------------------------------------------


def test_stimuli_load_in_sample_id_order(fake_synthetic):
    dataset = _build(fake_synthetic)
    ids = []
    first_batch = None
    for batch, batch_ids in dataset.load_videos(batch_size=8):
        if first_batch is None:
            first_batch = batch
        ids.extend(batch_ids)
    assert ids == dataset.sample_ids
    # (B, C, T, H, W) with a singleton time axis for still images.
    assert first_batch.ndim == 5
    assert first_batch.shape[:3] == (8, 3, 1)
    assert first_batch.dtype == torch.uint8


def test_missing_stimulus_names_the_preparation_step(fake_synthetic):
    (fake_synthetic["stimulus_path"] / "images" / "syn0001.png").unlink()
    dataset = _build(fake_synthetic)
    with pytest.raises(FileNotFoundError, match="prepare_stimuli"):
        next(iter(dataset.load_videos()))


def test_wrong_stimulus_directory_is_refused(fake_synthetic):
    """Guards the one mistake the ID namespace alone would not catch early."""
    (fake_synthetic["stimulus_path"] / PROVENANCE_FILENAME).write_text(
        json.dumps({"stimulus_set": NSD_STIMULUS_SET})
    )
    with pytest.raises(ValueError, match="wrong directory"):
        _build(fake_synthetic)


# --- fMRI alignment ---------------------------------------------------------


def test_fmri_rows_line_up_with_sample_ids(fake_synthetic):
    dataset = _build(fake_synthetic)
    data = dataset.load_fmri_data()

    assert set(data) == set(SUBJECTS)
    for subject_id in SUBJECTS:
        for roi_name in ROIS:
            betas = data[subject_id][roi_name]
            assert betas.shape == (len(dataset), VERTS[roi_name])
            # Every row carries its own stimulus id.
            assert np.array_equal(
                betas[:, 0], np.asarray(dataset.stimulus_ids, dtype=np.float32)
            )


def test_a_core_cache_reachable_by_the_superset_lookup_is_refused(
    fake_synthetic,
):
    """The sharpest edge: a 1000-row NSD-core cache under a superset ROI set.

    It is internally well-ordered, so without the stimulus-set tag it would
    pass every check, have its first 220 rows taken, and silently score the
    model against NSD-core brain responses.
    """
    core_counts = np.full(1000, 3, dtype=np.uint8)
    write_cache(
        fake_synthetic["roi_beta_cache"],
        "subjA",
        core_counts,
        fake_synthetic["n_vertices"],
        rois=["V1", "MT"],
        stimulus_set=NSD_STIMULUS_SET,
        dirname=NSD_SYNTHETIC_CACHE_DIRNAME,
    )
    # Same ROI set, so this is the exact-match path rather than the glob.
    dataset = _build(fake_synthetic, subject_ids=["subjA"])
    with pytest.raises(ValueError, match="stimulus set"):
        dataset.load_fmri_data()


def test_a_cache_with_the_wrong_number_of_stimuli_is_refused(fake_synthetic):
    write_cache(
        fake_synthetic["roi_beta_cache"],
        "subjA",
        fake_synthetic["design_counts"][:100],
        fake_synthetic["n_vertices"],
    )
    dataset = _build(fake_synthetic, subject_ids=["subjA"])
    with pytest.raises(ValueError, match="hold 100 stimuli"):
        dataset.load_fmri_data()


def test_tampered_trial_counts_are_caught(fake_synthetic):
    """The counts in the cache must match the published design."""
    target = cache_path(
        fake_synthetic["roi_beta_cache"],
        "subjA",
        ROIS,
        "fsaverage",
        6,
        dirname=NSD_SYNTHETIC_CACHE_DIRNAME,
    )
    with np.load(target, allow_pickle=False) as cached:
        contents = dict(cached)
    contents["trial_counts"] = contents["trial_counts"].copy()
    contents["trial_counts"][5] += 1
    np.savez_compressed(target, **contents)

    dataset = _build(fake_synthetic, subject_ids=["subjA"])
    with pytest.raises(ValueError, match="disagree with the counts"):
        dataset.load_fmri_data()


def test_a_declared_repetition_cap_is_honoured(fake_synthetic):
    """A cache built with --max-reps declares it, and the check adapts."""
    capped = np.minimum(fake_synthetic["design_counts"], 2).astype(np.uint8)
    write_cache(
        fake_synthetic["roi_beta_cache"],
        "subjA",
        capped,
        fake_synthetic["n_vertices"],
        max_reps=2,
    )
    dataset = _build(fake_synthetic, subject_ids=["subjA"])
    data = dataset.load_fmri_data()
    assert data["subjA"]["V1"].shape[0] == len(dataset)


def test_a_cap_the_cache_does_not_declare_is_caught(fake_synthetic):
    """Capped counts without the declaration must not pass silently."""
    capped = np.minimum(fake_synthetic["design_counts"], 2).astype(np.uint8)
    write_cache(
        fake_synthetic["roi_beta_cache"],
        "subjA",
        capped,
        fake_synthetic["n_vertices"],
        max_reps=-1,
    )
    dataset = _build(fake_synthetic, subject_ids=["subjA"])
    with pytest.raises(ValueError, match="disagree with the counts"):
        dataset.load_fmri_data()


def test_missing_cache_names_the_preparation_step(fake_synthetic):
    cache_path(
        fake_synthetic["roi_beta_cache"],
        "subjA",
        ROIS,
        "fsaverage",
        6,
        dirname=NSD_SYNTHETIC_CACHE_DIRNAME,
    ).unlink()
    dataset = _build(fake_synthetic, subject_ids=["subjA"])
    with pytest.raises(FileNotFoundError, match="nsd_synthetic.prepare_betas"):
        dataset.load_fmri_data()


# --- noise ceiling ----------------------------------------------------------


def test_voxel_consistency_matches_the_roi_widths(fake_synthetic):
    dataset = _build(fake_synthetic)
    ncsnr = dataset.load_voxel_consistency("subjA")
    assert set(ncsnr) == set(ROIS)
    for roi_name in ROIS:
        assert ncsnr[roi_name].shape == (VERTS[roi_name],)


def test_ncsnr_source_selects_between_the_two_maps(fake_synthetic):
    synthetic = _build(fake_synthetic).load_voxel_consistency("subjA")
    core = _build(
        fake_synthetic,
        ncsnr_source="nsd_core",
        nsd_core_fmri_path=fake_synthetic["nsd_core_fmri_path"],
    ).load_voxel_consistency("subjA")
    # The fixture writes the core map at half the scale.
    assert not np.allclose(synthetic["V1"], core["V1"])
    assert np.allclose(core["V1"], synthetic["V1"] * 0.5)


def test_nsd_core_source_needs_the_core_path(fake_synthetic):
    with pytest.raises(ValueError, match="nsd_core_fmri_path"):
        _build(fake_synthetic, ncsnr_source="nsd_core")


def test_unknown_ncsnr_source_is_rejected(fake_synthetic):
    with pytest.raises(ValueError, match="Unknown ncsnr_source"):
        _build(fake_synthetic, ncsnr_source="made_up")


def test_missing_synthetic_ncsnr_says_it_is_not_published(fake_synthetic):
    dataset = _build(fake_synthetic)
    path = (
        fake_synthetic["fmri_path"]
        / "nsddata_betas"
        / "ppdata"
        / "subjA"
        / "fsaverage"
        / NSD_SYNTHETIC_BETAS_DIRNAME
        / "prepared_betas"
        / "subjA_ncsnr.npz"
    )
    path.unlink()
    with pytest.raises(FileNotFoundError, match="publishes no noise ceiling"):
        dataset.load_voxel_consistency("subjA")


# --- real data --------------------------------------------------------------


@requires_real_data
def test_real_nsd_synthetic_loads():
    dataset = _real_dataset()
    assert len(dataset) == NSD_SYNTHETIC_N_ACHROMATIC
    assert dataset.sample_ids[0] == "syn0001"
    assert dataset.subject_ids == list(NSD_SUBJECTS)


@requires_real_data
def test_real_nsd_synthetic_fmri_alignment():
    dataset = _real_dataset(subject_ids=["subj01", "subj04"])
    data = dataset.load_fmri_data()
    for subject_id in ("subj01", "subj04"):
        for roi_name, betas in data[subject_id].items():
            assert betas.shape[0] == NSD_SYNTHETIC_N_ACHROMATIC
            assert np.isfinite(betas).all(), roi_name


@requires_real_data
def test_real_nsd_synthetic_stimuli_are_square():
    dataset = _real_dataset(subject_ids=["subj01"], resolution=256)
    batch, ids = next(iter(dataset.load_videos(batch_size=4)))
    assert ids == dataset.sample_ids[:4]
    assert batch.shape[-2:] == (256, 256)


def _real_dataset(**kwargs):
    from multitasking.CONSTANTS import NSD_ROIS
    from multitasking.datasets_nsd_synthetic import NSDSynthetic

    return NSDSynthetic(
        stimulus_path=STIMULUS_ROOT,
        fmri_path=SYNTHETIC_ROOT / "raw",
        rois=list(NSD_ROIS),
        split="test",
        atlas_dir=NSD_CORE_ROOT / "atlas",
        roi_beta_cache=BETA_CACHE,
        **kwargs,
    )
