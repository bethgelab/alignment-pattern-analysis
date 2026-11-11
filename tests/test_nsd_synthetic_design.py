"""Tests for the NSD-synthetic experiment design.

These are pure: the derivations under test read a ``.mat`` file and some
integers, so most of them run against a hand-built ``masterordering``. The
golden values at the bottom pin the real design, and skip when the checked-in
counts or the design file are not on the machine.
"""

import collections
from pathlib import Path

import numpy as np
import pytest

from multitasking.nsd_synthetic.design import (
    NSD_SYNTHETIC_N_ACHROMATIC,
    NSD_SYNTHETIC_N_IMAGES,
    NSD_SYNTHETIC_N_TRIALS,
    TRIAL_COUNTS_PATH,
    achromatic_stimulus_ids,
    compute_trial_counts,
    format_sample_id,
    load_masterordering,
    load_trial_counts,
    repetitions_by_image,
)

EXPDESIGN_PATH = Path(
    "/mnt/lustre/work/bethge/bkr578/data/NSD-synthetic/raw/nsddata/experiments/"
    "nsdsynthetic/nsdsynthetic_expdesign.mat"
)

#: The real repetition histogram of the 220 scored stimuli. Four scene-derived
#: subclasses are shown far more often than the rest, which is what makes the
#: distribution lopsided.
EXPECTED_ACHROMATIC_HISTOGRAM = {2: 181, 4: 23, 8: 8, 10: 8}

#: And of all 284, including the chromatic images this repository does not use.
EXPECTED_FULL_HISTOGRAM = {2: 236, 4: 32, 8: 8, 10: 8}


def _fake_master_ordering(extra_reps: dict[int, int] | None = None):
    """Builds a trial sequence showing every image twice, plus extras.

    Args:
        extra_reps: ``{image_id: additional presentations}``.

    Returns:
        A 1-based trial-to-image array.
    """
    trials = list(range(1, NSD_SYNTHETIC_N_IMAGES + 1)) * 2
    for image_id, extra in (extra_reps or {}).items():
        trials.extend([image_id] * extra)
    return np.asarray(trials, dtype=np.int64)


# --- sample IDs -------------------------------------------------------------


def test_sample_ids_are_namespaced_away_from_nsd_core():
    """The whole point: these must not collide with NSD-core's "0001".

    NSD-core stimuli 1-220 are named "0001".."0220" and their PNGs live in a
    sibling directory, so a shared namespace would let a mis-pointed stimulus
    path pair natural images with synthetic betas undetected.
    """
    from multitasking.nsd.splits import format_sample_id as nsd_format

    for stimulus_id in (1, 42, 220):
        assert format_sample_id(stimulus_id) != nsd_format(stimulus_id)
        assert format_sample_id(stimulus_id).startswith("syn")


def test_sample_ids_sort_in_stimulus_order():
    ids = [format_sample_id(i) for i in achromatic_stimulus_ids()]
    assert ids == sorted(ids)
    assert ids[0] == "syn0001"
    assert ids[-1] == "syn0220"


def test_achromatic_ids_are_the_shared_prefix():
    ids = achromatic_stimulus_ids()
    assert ids == list(range(1, NSD_SYNTHETIC_N_ACHROMATIC + 1))
    assert len(ids) == NSD_SYNTHETIC_N_ACHROMATIC
    assert max(ids) < NSD_SYNTHETIC_N_IMAGES


# --- trial counts -----------------------------------------------------------


def test_trial_counts_count_presentations():
    counts = compute_trial_counts(_fake_master_ordering({7: 3, 9: 1}))
    assert len(counts) == NSD_SYNTHETIC_N_IMAGES
    assert counts[6] == 5  # image 7: two plus three
    assert counts[8] == 3  # image 9: two plus one
    assert counts[0] == 2


def test_trial_counts_reject_a_design_with_unrepeated_images():
    """Every NSD-synthetic image is shown at least once per task."""
    trials = np.asarray(
        list(range(1, NSD_SYNTHETIC_N_IMAGES + 1)) * 2, dtype=np.int64
    )
    trials[0] = 2  # image 1 now appears only once
    with pytest.raises(ValueError, match="fewer than twice"):
        compute_trial_counts(trials)


def test_trial_counts_do_not_depend_on_the_subject():
    """All eight subjects completed the single session, unlike NSD-core."""
    master_ordering = _fake_master_ordering({13: 8})
    assert np.array_equal(
        compute_trial_counts(master_ordering),
        compute_trial_counts(master_ordering),
    )


# --- repetition grouping ----------------------------------------------------


def test_repetitions_partition_the_trials():
    master_ordering = _fake_master_ordering({3: 6, 11: 2})
    groups = repetitions_by_image(master_ordering)

    assert len(groups) == NSD_SYNTHETIC_N_IMAGES
    assert sorted(np.concatenate(groups).tolist()) == list(
        range(len(master_ordering))
    )
    for image_id, trials in enumerate(groups, start=1):
        assert set(master_ordering[trials].tolist()) == {image_id}


def test_repetitions_are_ascending():
    groups = repetitions_by_image(_fake_master_ordering({5: 4}))
    for trials in groups:
        assert np.array_equal(trials, np.sort(trials))


def test_max_reps_keeps_the_earliest():
    master_ordering = _fake_master_ordering({3: 6})
    uncapped = repetitions_by_image(master_ordering)
    capped = repetitions_by_image(master_ordering, max_reps=2)

    assert len(capped[2]) == 2
    assert np.array_equal(capped[2], uncapped[2][:2])
    # Images at the minimum are untouched.
    assert np.array_equal(capped[0], uncapped[0])


def test_max_reps_below_one_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        repetitions_by_image(_fake_master_ordering(), max_reps=0)


# --- the real design --------------------------------------------------------


@pytest.mark.skipif(
    not TRIAL_COUNTS_PATH.exists(),
    reason="cached NSD-synthetic trial counts not generated yet",
)
def test_cached_counts_match_the_published_design():
    counts = load_trial_counts()
    assert len(counts) == NSD_SYNTHETIC_N_IMAGES
    assert int(counts.sum()) == NSD_SYNTHETIC_N_TRIALS
    assert int(counts.min()) == 2

    histogram = dict(sorted(collections.Counter(counts.tolist()).items()))
    assert histogram == EXPECTED_FULL_HISTOGRAM

    achromatic = counts[:NSD_SYNTHETIC_N_ACHROMATIC]
    assert dict(
        sorted(collections.Counter(achromatic.tolist()).items())
    ) == EXPECTED_ACHROMATIC_HISTOGRAM
    assert int(achromatic.sum()) == 598


@pytest.mark.skipif(
    not EXPDESIGN_PATH.exists(),
    reason="nsdsynthetic_expdesign.mat not available on this machine",
)
def test_cached_counts_match_freshly_computed():
    """Guards against the checked-in cache drifting from the design file."""
    fresh = compute_trial_counts(load_masterordering(EXPDESIGN_PATH))
    assert np.array_equal(fresh, load_trial_counts())


@pytest.mark.skipif(
    not EXPDESIGN_PATH.exists(),
    reason="nsdsynthetic_expdesign.mat not available on this machine",
)
def test_master_ordering_describes_one_session():
    master_ordering = load_masterordering(EXPDESIGN_PATH)
    assert master_ordering.shape == (NSD_SYNTHETIC_N_TRIALS,)
    assert len(np.unique(master_ordering)) == NSD_SYNTHETIC_N_IMAGES
