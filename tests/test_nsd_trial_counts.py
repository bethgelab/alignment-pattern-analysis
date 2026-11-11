"""Tests for deriving NSD shared-1000 repetition counts."""

from pathlib import Path

import numpy as np
import pytest

from multitasking.nsd.trial_counts import (
    NSD_SESSIONS_PER_SUBJECT,
    NSD_TRIALS_PER_SESSION,
    compute_shared1000_trial_counts,
    load_shared1000_trial_counts,
    valid_stimulus_ids,
)

EXPDESIGN_PATH = Path(
    "/mnt/lustre/work/bethge/bkr857/projects/metrics_benchmark/nsd_expdesign.mat"
)

# Counts derived from the real experiment design, by completed session count.
# metrics_benchmark documents 827 for a 32-session subject at min_n_trials=2,
# which is what pins this derivation to the reference implementation.
EXPECTED_BY_SESSIONS = {
    40: {1: 1000, 2: 1000, 3: 1000},
    32: {1: 930, 2: 827, 3: 614},
    30: {1: 907, 2: 766, 3: 515},
}

# Intersections over all eight subjects.
EXPECTED_INTERSECTION = {1: 907, 2: 766, 3: 515}


def test_cached_counts_match_expected_per_session_count():
    counts = load_shared1000_trial_counts()
    for subject_id, n_sessions in NSD_SESSIONS_PER_SUBJECT.items():
        expected = EXPECTED_BY_SESSIONS[n_sessions]
        for min_n_trials, expected_n in expected.items():
            actual = int((counts[subject_id] >= min_n_trials).sum())
            assert actual == expected_n, (
                f"{subject_id} ({n_sessions} sessions) at "
                f"min_n_trials={min_n_trials}: {actual} != {expected_n}"
            )


def test_derivation_on_synthetic_master_ordering(monkeypatch, tmp_path):
    """Counts come from masterordering alone; verify with a hand-built one."""
    n_sessions = 2
    master = np.zeros(NSD_TRIALS_PER_SESSION * n_sessions, dtype=np.uint16)
    # Stimulus 1 three times, stimulus 2 twice, stimulus 3 once.
    master[0:3] = 1
    master[3:5] = 2
    master[5:6] = 3
    # Everything else is a subject-unique stimulus, i.e. above 1000.
    master[6:] = 5000

    # compute_shared1000_trial_counts imports loadmat lazily from scipy.io,
    # so that is where the patch has to land.
    import scipy.io

    monkeypatch.setattr(
        scipy.io, "loadmat", lambda _: {"masterordering": master[None, :]}
    )

    expdesign = tmp_path / "nsd_expdesign.mat"
    expdesign.touch()
    counts = compute_shared1000_trial_counts(expdesign, {"subjX": n_sessions})

    assert counts["subjX"][0] == 3
    assert counts["subjX"][1] == 2
    assert counts["subjX"][2] == 1
    assert counts["subjX"][3:].sum() == 0

    assert valid_stimulus_ids(["subjX"], 1, counts) == [1, 2, 3]
    assert valid_stimulus_ids(["subjX"], 2, counts) == [1, 2]
    assert valid_stimulus_ids(["subjX"], 3, counts) == [1]


@pytest.mark.skipif(
    not EXPDESIGN_PATH.exists(), reason="nsd_expdesign.mat not available"
)
def test_cached_counts_match_freshly_computed():
    """The checked-in cache must not drift from the experiment design."""
    fresh = compute_shared1000_trial_counts(EXPDESIGN_PATH)
    cached = load_shared1000_trial_counts()
    assert sorted(fresh) == sorted(cached)
    for subject_id in fresh:
        np.testing.assert_array_equal(fresh[subject_id], cached[subject_id])


