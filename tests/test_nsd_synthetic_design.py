"""Tests for the NSD-synthetic experiment design.

The repetition grouping runs against a hand-built ``masterordering``; the
drift check against the real design file skips when it is not on the machine.
"""

from pathlib import Path

import numpy as np
import pytest

from multitasking.nsd_synthetic.design import (
    NSD_SYNTHETIC_N_IMAGES,
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


def test_repetitions_partition_the_trials():
    """Every trial is averaged into exactly one image, and the right one."""
    master_ordering = _fake_master_ordering({3: 6, 11: 2})
    groups = repetitions_by_image(master_ordering)

    assert len(groups) == NSD_SYNTHETIC_N_IMAGES
    assert sorted(np.concatenate(groups).tolist()) == list(
        range(len(master_ordering))
    )
    for image_id, trials in enumerate(groups, start=1):
        assert set(master_ordering[trials].tolist()) == {image_id}


@pytest.mark.skipif(
    not EXPDESIGN_PATH.exists(),
    reason="nsdsynthetic_expdesign.mat not available on this machine",
)
def test_cached_counts_match_freshly_computed():
    """Guards against the checked-in cache drifting from the design file."""
    fresh = compute_trial_counts(load_masterordering(EXPDESIGN_PATH))
    assert np.array_equal(fresh, load_trial_counts())
