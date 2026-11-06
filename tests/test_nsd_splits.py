"""Tests for the deterministic NSD train/test split."""

import sys

import pytest

from multitasking.nsd.splits import (
    format_sample_id,
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


def test_default_fraction_is_eighty_twenty():
    splits = split_shared1000_ids(0.8)
    assert len(splits["train"]) == 800
    assert len(splits["test"]) == 200


@pytest.mark.parametrize(
    "train_fraction,n_train", [(0.5, 500), (0.75, 750), (0.8, 800), (0.9, 900)]
)
def test_fraction_is_honoured(train_fraction: float, n_train: int):
    splits = split_shared1000_ids(train_fraction)
    assert len(splits["train"]) == n_train
    assert len(splits["test"]) == NSD_SHARED1000_SIZE - n_train


def test_splits_are_sorted():
    splits = split_shared1000_ids()
    for ids in splits.values():
        assert ids == sorted(ids)


def test_split_is_deterministic_across_calls():
    assert split_shared1000_ids() == split_shared1000_ids()


def test_split_uses_no_randomness(monkeypatch):
    """A seeded RNG anywhere would make the split irreproducible."""
    import random

    def explode(*args, **kwargs):
        raise AssertionError("the split must not use randomness")

    monkeypatch.setattr(random, "random", explode)
    monkeypatch.setattr(random, "shuffle", explode)
    monkeypatch.setattr(random, "sample", explode)
    if "numpy.random" in sys.modules:
        import numpy as np

        monkeypatch.setattr(np.random, "permutation", explode)
        monkeypatch.setattr(np.random, "shuffle", explode)

    split_shared1000_ids()


@pytest.mark.parametrize("train_fraction", [0.0, 1.0, -0.5, 1.5])
def test_out_of_range_fraction_raises(train_fraction: float):
    with pytest.raises(ValueError, match="train_fraction"):
        split_shared1000_ids(train_fraction)


def test_fraction_below_half_raises():
    with pytest.raises(ValueError, match="half"):
        split_shared1000_ids(0.3)


def test_fraction_without_integer_stride_raises():
    with pytest.raises(ValueError, match="integer stride"):
        split_shared1000_ids(0.7)


def test_unknown_split_raises():
    with pytest.raises(ValueError, match="Unknown split"):
        split_sample_ids("validation", [1, 2, 3])


def test_split_sample_ids_intersects_with_available():
    available = [1, 2, 3, 4, 5, 10, 15, 20]
    train = split_sample_ids("train", available)
    test = split_sample_ids("test", available)

    assert set(train) | set(test) == set(available)
    assert set(train).isdisjoint(test)
    # Every 5th stimulus is held out at the default fraction.
    assert test == [5, 10, 15, 20]


def test_membership_does_not_depend_on_subject_set():
    """The property the whole design hinges on: a stimulus never swaps split."""
    counts = load_shared1000_trial_counts()
    all_subjects = sorted(NSD_SESSIONS_PER_SUBJECT)
    forty_session = [s for s, n in NSD_SESSIONS_PER_SUBJECT.items() if n == 40]

    reference = split_shared1000_ids()
    for subjects in (all_subjects, forty_session, ["subj04"], ["subj01", "subj02"]):
        valid = valid_stimulus_ids(subjects, 2, counts)
        for split in ("train", "test"):
            ids = split_sample_ids(split, valid)
            assert set(ids) <= set(reference[split])


def test_membership_does_not_depend_on_min_n_trials():
    counts = load_shared1000_trial_counts()
    all_subjects = sorted(NSD_SESSIONS_PER_SUBJECT)

    reference = split_shared1000_ids()
    for min_n_trials in (1, 2, 3):
        valid = valid_stimulus_ids(all_subjects, min_n_trials, counts)
        for split in ("train", "test"):
            assert set(split_sample_ids(split, valid)) <= set(reference[split])


def test_realized_counts_for_the_default_configuration():
    """Documents the numbers a default NSD run will actually report."""
    counts = load_shared1000_trial_counts()
    valid = valid_stimulus_ids(sorted(NSD_SESSIONS_PER_SUBJECT), 2, counts)

    assert len(valid) == 766
    assert len(split_sample_ids("train", valid)) == 618
    assert len(split_sample_ids("test", valid)) == 148


def test_forty_session_subjects_give_exactly_eight_hundred():
    counts = load_shared1000_trial_counts()
    forty_session = [s for s, n in NSD_SESSIONS_PER_SUBJECT.items() if n == 40]
    valid = valid_stimulus_ids(forty_session, 2, counts)

    assert len(split_sample_ids("train", valid)) == 800
    assert len(split_sample_ids("test", valid)) == 200


@pytest.mark.parametrize(
    "stimulus_id,expected", [(1, "0001"), (42, "0042"), (1000, "1000")]
)
def test_format_sample_id(stimulus_id: int, expected: str):
    assert format_sample_id(stimulus_id) == expected
