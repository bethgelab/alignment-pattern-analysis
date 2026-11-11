"""Tests for the deterministic NSD train/test split."""

from multitasking.nsd.splits import (
    split_sample_ids,
    split_shared1000_ids,
)
from multitasking.nsd.trial_counts import (
    NSD_SESSIONS_PER_SUBJECT,
    NSD_SHARED1000_SIZE,
    load_shared1000_trial_counts,
    valid_stimulus_ids,
)


def test_splits_partition_all_stimuli():
    splits = split_shared1000_ids()
    train, test = splits["train"], splits["test"]
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == list(range(1, NSD_SHARED1000_SIZE + 1))


def test_membership_does_not_depend_on_subject_set_or_min_n_trials():
    """The property the design hinges on: a stimulus never swaps split."""
    counts = load_shared1000_trial_counts()
    all_subjects = sorted(NSD_SESSIONS_PER_SUBJECT)
    forty_session = [s for s, n in NSD_SESSIONS_PER_SUBJECT.items() if n == 40]
    reference = split_shared1000_ids()

    configs = [
        (subjects, min_n)
        for subjects in (all_subjects, forty_session, ["subj04"], ["subj01", "subj02"])
        for min_n in (1, 2, 3)
    ]
    for subjects, min_n in configs:
        valid = set(valid_stimulus_ids(subjects, min_n, counts))
        for split in ("train", "test"):
            expected = [i for i in reference[split] if i in valid]
            assert split_sample_ids(split, sorted(valid)) == expected


def test_realized_counts_for_the_default_configuration():
    """Documents the numbers a default NSD run will actually report."""
    counts = load_shared1000_trial_counts()
    valid = valid_stimulus_ids(sorted(NSD_SESSIONS_PER_SUBJECT), 2, counts)

    assert len(valid) == 766
    assert len(split_sample_ids("train", valid)) == 618
    assert len(split_sample_ids("test", valid)) == 148


