"""The NSD-synthetic experiment design.

NSD-synthetic adds one scan session per subject to the eight NSD-core subjects,
showing 284 deliberately non-naturalistic images across 744 stimulus trials. Two
properties of that design shape everything downstream:

*The betas are per trial, not per image.* ``masterordering`` gives the image
shown on each of the 744 trials, with values in ``1..284``. Every image was
shown at least twice - once under the fixation task and once under the one-back
task - and the scene-derived subclasses up to ten times. Averaging a stimulus's
repetitions therefore requires the design file; the betas alone do not say which
trial belongs to which image.

*Only 220 of the 284 images are shared across subjects.* Images 1-220 are
achromatic and ship in a single ``nsdsynthetic_stimuli.hdf5``. Images 221-284
are chromatic and ship as eight per-subject renderings, one per subject's
monitor calibration. Model features are extracted once per split and reused for
every subject, so only the achromatic set can be used without breaking the
sample-ordering contract. :data:`NSD_SYNTHETIC_N_ACHROMATIC` is that boundary,
and it is a file boundary rather than an assumed index range.

Unlike NSD-core, the repetition counts do not vary by subject: every subject
completed the single session in full, so one count vector describes them all.
"""

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)

#: Total images in NSD-synthetic, achromatic and chromatic together.
NSD_SYNTHETIC_N_IMAGES = 284

#: Images shipped in the shared, subject-independent stimulus file. Stimulus IDs
#: ``1..220``; the remaining 64 are the per-subject chromatic renderings.
NSD_SYNTHETIC_N_ACHROMATIC = 220

#: Stimulus trials in the session, across all eight runs.
NSD_SYNTHETIC_N_TRIALS = 744

#: Stimulus runs in the session, alternating fixation and one-back task.
NSD_SYNTHETIC_N_RUNS = 8

#: Stimulus trials per run (the other 14 of the 107 trial slots are blank).
NSD_SYNTHETIC_TRIALS_PER_RUN = 93

#: Name recorded in prepared artifacts to identify this stimulus set. Asserted
#: when a cache is loaded, so that an NSD-core cache can never be mistaken for
#: an NSD-synthetic one.
NSD_SYNTHETIC_STIMULUS_SET = "nsd_synthetic_achromatic"

#: Cached counts, so runs need neither nsdsynthetic_expdesign.mat nor scipy.
TRIAL_COUNTS_PATH = (
    Path(__file__).parent / "metadata" / "nsdsynthetic_trial_counts.npz"
)

#: Key under which the counts live in that ``.npz``.
TRIAL_COUNTS_KEY = "trial_counts"


def achromatic_stimulus_ids() -> list[int]:
    """Returns the stimulus IDs of the shared achromatic images.

    Returns:
        ``[1, 2, ..., 220]``.
    """
    return list(range(1, NSD_SYNTHETIC_N_ACHROMATIC + 1))


def format_sample_id(stimulus_id: int) -> str:
    """Formats a stimulus ID the way the pipeline keys NSD-synthetic samples.

    Deliberately *not* the bare four-digit form NSD-core and BOLD Moments use.
    NSD-core stimuli 1-220 are already named ``"0001".."0220"`` and their PNGs
    sit in a sibling directory, so a shared namespace would let a mis-pointed
    stimulus path pair natural images with synthetic betas while every check
    still passed - the feature caches verify sample IDs, which are strings, not
    pixels.

    Args:
        stimulus_id: Stimulus ID in ``1..284``.

    Returns:
        For example ``"syn0042"``.
    """
    return f"syn{stimulus_id:04d}"


def load_masterordering(expdesign_path: str | Path) -> NDArray[np.int64]:
    """Loads the trial-to-image mapping from the experiment design.

    Args:
        expdesign_path: Path to ``nsdsynthetic_expdesign.mat``.

    Returns:
        A length-744 array of 1-based image IDs, in presentation order.

    Raises:
        FileNotFoundError: If the experiment design file is missing.
        ValueError: If its shape or value range is not what this design implies.
    """
    # Imported lazily so that loading the cached counts needs no scipy.
    from scipy.io import loadmat

    expdesign_path = Path(expdesign_path)
    if not expdesign_path.exists():
        raise FileNotFoundError(
            f"NSD-synthetic experiment design not found at {expdesign_path}. "
            "Download it with `./tasks/nsd_synthetic/download.sh`, or directly "
            "from https://natural-scenes-dataset.s3.amazonaws.com/nsddata/"
            "experiments/nsdsynthetic/nsdsynthetic_expdesign.mat"
        )

    master_ordering = np.asarray(
        loadmat(expdesign_path)["masterordering"]
    ).squeeze().astype(np.int64)

    if master_ordering.shape != (NSD_SYNTHETIC_N_TRIALS,):
        raise ValueError(
            f"Expected masterordering of shape ({NSD_SYNTHETIC_N_TRIALS},) in "
            f"{expdesign_path}, got {master_ordering.shape}."
        )
    low, high = int(master_ordering.min()), int(master_ordering.max())
    if low != 1 or high != NSD_SYNTHETIC_N_IMAGES:
        raise ValueError(
            f"masterordering in {expdesign_path} spans {low}..{high}, expected "
            f"1..{NSD_SYNTHETIC_N_IMAGES}."
        )

    return master_ordering


def compute_trial_counts(
    master_ordering: NDArray[np.int64],
) -> NDArray[np.uint8]:
    """Counts how often each image was shown, from the trial sequence.

    Args:
        master_ordering: The length-744 trial-to-image mapping.

    Returns:
        A length-284 array where entry ``i`` is how often image ``i + 1`` was
        shown. The same for every subject, since all eight completed the single
        session in full.

    Raises:
        ValueError: If any image was shown fewer than twice, which would mean
            the design file does not describe NSD-synthetic.
    """
    # bincount index i counts image id i, so drop the unused 0 slot.
    counts = np.bincount(
        master_ordering, minlength=NSD_SYNTHETIC_N_IMAGES + 1
    )[1 : NSD_SYNTHETIC_N_IMAGES + 1]

    if (counts < 2).any():
        too_few = (np.flatnonzero(counts < 2) + 1).tolist()
        raise ValueError(
            f"Images {too_few[:10]} were shown fewer than twice. Every "
            "NSD-synthetic image is shown at least once per task, so this "
            "design file is not the one this code expects."
        )

    return counts.astype(np.uint8)


def repetitions_by_image(
    master_ordering: NDArray[np.int64], max_reps: int | None = None
) -> list[NDArray[np.int64]]:
    """Groups trial indices by the image shown on them.

    Args:
        master_ordering: The length-744 trial-to-image mapping.
        max_reps: Keep at most this many repetitions per image, the earliest
            ones. ``None`` keeps all of them.

    Returns:
        A list of length 284; entry ``i`` holds the trial indices on which image
        ``i + 1`` was shown, ascending.

    Raises:
        ValueError: If ``max_reps`` is less than 1.
    """
    if max_reps is not None and max_reps < 1:
        raise ValueError(f"max_reps must be at least 1, got {max_reps}")

    order = np.argsort(master_ordering, kind="stable")
    boundaries = np.searchsorted(
        master_ordering[order], np.arange(1, NSD_SYNTHETIC_N_IMAGES + 2)
    )
    groups = [
        order[boundaries[i] : boundaries[i + 1]]
        for i in range(NSD_SYNTHETIC_N_IMAGES)
    ]
    if max_reps is not None:
        groups = [group[:max_reps] for group in groups]
    return groups


def load_trial_counts(path: str | Path | None = None) -> NDArray[np.uint8]:
    """Loads the cached per-image repetition counts.

    Args:
        path: Location of the cached ``.npz``. Defaults to
            :data:`TRIAL_COUNTS_PATH`.

    Returns:
        A length-284 array, as :func:`compute_trial_counts`.

    Raises:
        FileNotFoundError: If the cache is missing, naming the script that
            regenerates it.
        ValueError: If the cache does not hold the expected array.
    """
    path = Path(path) if path is not None else TRIAL_COUNTS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Cached NSD-synthetic trial counts not found at {path}. "
            "Regenerate them with `python -m "
            "multitasking.nsd_synthetic.prepare_trial_counts --expdesign "
            "<nsdsynthetic_expdesign.mat>`."
        )
    with np.load(path) as cached:
        if TRIAL_COUNTS_KEY not in cached:
            raise ValueError(
                f"{path} is missing {TRIAL_COUNTS_KEY!r}; found {list(cached)}."
            )
        counts = cached[TRIAL_COUNTS_KEY]

    if counts.shape != (NSD_SYNTHETIC_N_IMAGES,):
        raise ValueError(
            f"{path} holds counts of shape {counts.shape}, expected "
            f"({NSD_SYNTHETIC_N_IMAGES},)."
        )
    return counts
