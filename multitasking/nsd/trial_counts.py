"""How often each shared-1000 stimulus was shown to each NSD subject.

NSD subjects completed different numbers of scan sessions (30, 32 or 40), so a
stimulus that one subject saw three times another may have seen only once, or
not at all. Any analysis that compares subjects therefore has to agree on which
stimuli to use.

The repetition counts follow entirely from the experiment design: a subject who
completed ``n`` sessions saw the first ``750 * n`` entries of ``masterordering``,
and entries with a value of 1000 or less are the shared-1000 stimuli. So the
counts can be derived without reading a single beta file - which matters,
because the prepared betas are about 3.9 GB per subject and the stimulus set
has to be fixed before any of them are opened.

The derivation is cheap but it is still a derivation, so
``multitasking.datasets_nsd`` re-checks it against the NaN pattern of the betas
it actually loads.
"""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

#: Number of stimuli shown to every subject (NSD's "shared1000" set).
NSD_SHARED1000_SIZE = 1000

#: Stimulus trials per scan session.
NSD_TRIALS_PER_SESSION = 750

#: Completed scan sessions per subject, from the NSD data manual.
NSD_SESSIONS_PER_SUBJECT: dict[str, int] = {
    "subj01": 40,
    "subj02": 40,
    "subj03": 32,
    "subj04": 30,
    "subj05": 40,
    "subj06": 32,
    "subj07": 40,
    "subj08": 30,
}

#: Cached counts, so runs need neither nsd_expdesign.mat nor the betas.
TRIAL_COUNTS_PATH = Path(__file__).parent / "metadata" / "nsd_shared1000_trial_counts.npz"


def compute_shared1000_trial_counts(
    expdesign_path: str | Path,
    sessions_per_subject: Mapping[str, int] | None = None,
) -> dict[str, np.ndarray]:
    """Derives per-subject repetition counts from the NSD experiment design.

    Args:
        expdesign_path: Path to ``nsd_expdesign.mat``.
        sessions_per_subject: Completed sessions per subject. Defaults to
            :data:`NSD_SESSIONS_PER_SUBJECT`.

    Returns:
        ``{subject_id: counts}``, where ``counts`` has length
        :data:`NSD_SHARED1000_SIZE` and ``counts[i]`` is how often stimulus
        ``i + 1`` was shown to that subject.

    Raises:
        FileNotFoundError: If the experiment design file is missing.
    """
    # Imported lazily so that loading the cached counts needs no scipy.
    from scipy.io import loadmat

    expdesign_path = Path(expdesign_path)
    if not expdesign_path.exists():
        raise FileNotFoundError(
            f"NSD experiment design not found at {expdesign_path}. It can be "
            "downloaded from https://natural-scenes-dataset.s3.amazonaws.com/"
            "nsddata_other/experimentcode/stimulusfiles/nsd_expdesign.mat"
        )

    if sessions_per_subject is None:
        sessions_per_subject = NSD_SESSIONS_PER_SUBJECT

    master_ordering = loadmat(expdesign_path)["masterordering"].squeeze()

    counts = {}
    for subject_id, n_sessions in sessions_per_subject.items():
        seen = master_ordering[: NSD_TRIALS_PER_SESSION * n_sessions]
        shared = seen[seen <= NSD_SHARED1000_SIZE]
        # bincount index i counts stimulus id i, so drop the unused 0 slot.
        counts[subject_id] = np.bincount(
            shared, minlength=NSD_SHARED1000_SIZE + 1
        )[1 : NSD_SHARED1000_SIZE + 1].astype(np.uint8)

    return counts


def load_shared1000_trial_counts(
    path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Loads the cached per-subject repetition counts.

    Args:
        path: Location of the cached ``.npz``. Defaults to
            :data:`TRIAL_COUNTS_PATH`.

    Returns:
        ``{subject_id: counts}``, as :func:`compute_shared1000_trial_counts`.

    Raises:
        FileNotFoundError: If the cache is missing, naming the script that
            regenerates it.
    """
    path = Path(path) if path is not None else TRIAL_COUNTS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Cached NSD trial counts not found at {path}. Regenerate them with "
            "`python -m multitasking.nsd.prepare_trial_counts "
            "--expdesign <nsd_expdesign.mat>`."
        )
    with np.load(path) as cached:
        return {subject_id: cached[subject_id] for subject_id in cached.files}


def valid_stimulus_ids(
    subject_ids: Sequence[str],
    min_n_trials: int = 2,
    counts: Mapping[str, np.ndarray] | None = None,
) -> list[int]:
    """Returns the stimuli every given subject saw at least ``min_n_trials`` times.

    Args:
        subject_ids: Subjects that must all have seen the stimulus.
        min_n_trials: Required repetitions per subject.
        counts: Per-subject repetition counts. Loaded from the cache if None.

    Returns:
        Sorted, ascending stimulus IDs in ``1..1000``.

    Raises:
        ValueError: If no subjects are given, ``min_n_trials`` is below 1, or a
            subject has no counts.
    """
    if not subject_ids:
        raise ValueError("At least one subject is required.")
    if min_n_trials < 1:
        raise ValueError(f"min_n_trials must be at least 1, got {min_n_trials}")

    if counts is None:
        counts = load_shared1000_trial_counts()

    missing = [subject_id for subject_id in subject_ids if subject_id not in counts]
    if missing:
        raise ValueError(
            f"No trial counts for {missing}; known subjects: {sorted(counts)}"
        )

    # Session sequences are prefixes of one another, so these sets are nested
    # and the intersection equals the lowest-session subject's set. Intersecting
    # explicitly keeps that an observation rather than an assumption.
    valid = np.ones(NSD_SHARED1000_SIZE, dtype=bool)
    for subject_id in subject_ids:
        valid &= counts[subject_id] >= min_n_trials

    return (np.flatnonzero(valid) + 1).tolist()
