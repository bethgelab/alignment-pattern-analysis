"""Deterministic train/test split of NSD's shared-1000 stimuli.

NSD ships no train/test split, but the benchmark needs one: linear
predictivity is fit on train and evaluated on test, and RSA is computed per
split. The split here is a pure function of the stimulus ID - it depends on
neither the subject set, nor ``min_n_trials``, nor any RNG.

That independence is the point. Which stimuli *survive* does depend on the
subject set (subjects completed different numbers of sessions), so if the split
were taken after intersecting survivors, a stimulus could land in train for an
eight-subject benchmark run and in test for a two-subject pairwise run. Model
alignment and the brain-brain reference it is compared against would then be
computed on differently-split data. Splitting first and intersecting afterwards
costs a few stimuli per split and removes that failure mode entirely.
"""

from multitasking.nsd.trial_counts import NSD_SHARED1000_SIZE


def split_shared1000_ids(train_fraction: float = 0.8) -> dict[str, list[int]]:
    """Partitions stimulus IDs 1..1000 into a train and a test split.

    Every ``k``-th stimulus goes to test, where ``k = round(1 / (1 -
    train_fraction))``. Interleaving keeps the ratio approximately balanced
    within *any* subset of the IDs, which matters because the splits are later
    intersected with the stimuli a given subject set actually saw.

    Args:
        train_fraction: Fraction of stimuli assigned to train. Must be strictly
            between 0 and 1, and must correspond to an integer stride (0.8 is
            every 5th, 0.75 every 4th).

    Returns:
        ``{"train": [...], "test": [...]}``, each sorted ascending, together
        partitioning ``1..1000``.

    Raises:
        ValueError: If ``train_fraction`` is out of range or does not
            correspond to a usable integer stride.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            f"train_fraction must be strictly between 0 and 1, got {train_fraction}"
        )

    stride = round(1.0 / (1.0 - train_fraction))
    if stride < 2:
        raise ValueError(
            f"train_fraction={train_fraction} implies keeping fewer than half "
            "the stimuli for training; use a value of 0.5 or above."
        )

    realized = 1.0 - 1.0 / stride
    if abs(realized - train_fraction) > 1e-9:
        raise ValueError(
            f"train_fraction={train_fraction} does not correspond to an integer "
            f"stride; the nearest achievable fraction is {realized:.6f} "
            f"(every {stride}th stimulus held out)."
        )

    all_ids = range(1, NSD_SHARED1000_SIZE + 1)
    test = [stimulus_id for stimulus_id in all_ids if stimulus_id % stride == 0]
    train = [stimulus_id for stimulus_id in all_ids if stimulus_id % stride != 0]
    return {"train": train, "test": test}


def split_sample_ids(
    split: str,
    valid_ids: list[int],
    train_fraction: float = 0.8,
) -> list[int]:
    """Returns the stimuli of one split that the given subjects actually saw.

    Splits first, then intersects with ``valid_ids``; see the module docstring
    for why the order matters.

    Args:
        split: "train" or "test".
        valid_ids: Stimulus IDs available for the configured subject set, from
            ``multitasking.nsd.trial_counts.valid_stimulus_ids``.
        train_fraction: Fraction of stimuli assigned to train.

    Returns:
        Sorted, ascending stimulus IDs for the split.

    Raises:
        ValueError: If ``split`` is not "train" or "test".
    """
    splits = split_shared1000_ids(train_fraction)
    if split not in splits:
        raise ValueError(f"Unknown split {split!r}; expected 'train' or 'test'.")

    available = set(valid_ids)
    return [stimulus_id for stimulus_id in splits[split] if stimulus_id in available]


def format_sample_id(stimulus_id: int) -> str:
    """Formats a stimulus ID the way the pipeline keys samples.

    Zero-padded to four digits, matching the BOLD Moments convention, so that
    the cached-feature sample-ID checks compare like with like after a JSON
    round trip.

    Args:
        stimulus_id: Stimulus ID in ``1..1000``.

    Returns:
        The zero-padded string, e.g. ``"0042"``.
    """
    return f"{stimulus_id:04d}"
