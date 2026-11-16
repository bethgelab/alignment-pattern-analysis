"""Canonical ROI and subject lists used across the analyses.

``ROIS`` and ``SUBJECTS`` describe the BOLD Moments dataset and are the values
the paper's figures were produced with. They are imported directly by a dozen
analysis and plotting scripts, and are used as default arguments, so they must
not change. Dataset-aware code should use :func:`get_rois` and
:func:`get_subjects` instead.
"""

from collections.abc import Iterable
from dataclasses import dataclass

ROIS = [
    "V1", "V2", "V3", "V4", "V8", "PIT", "FFC", "V3A", "V3B", "V6",
    "V6A", "V7", "IPS1", "MT", "MST", "FST", "LO1", "LO2", "LO3",
]
SUBJECTS = [
    "sub-01", "sub-02", "sub-03", "sub-04", "sub-05",
    "sub-06", "sub-07", "sub-08", "sub-09", "sub-10",
]

# NSD uses the same HCP-MMP parcel names as BOLD Moments (verified: all of the
# ROIs in configs/benchmark.yaml are present in both hemispheres of the
# fsaverage HCP-MMP1 annotation), which is what keeps the cross-region
# alignment patterns comparable across the two datasets. It is kept as a
# separate symbol so the two lists can diverge without touching ROIS.
NSD_ROIS = list(ROIS)

# NSD ships eight subjects, named differently from BOLD Moments' "sub-NN".
NSD_SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]

BOLD_MOMENTS = "bold_moments"
NSD = "nsd"
# NSD-synthetic: the same eight subjects, the same fsaverage HCP-MMP parcels,
# 220 non-naturalistic stimuli in place of NSD's natural scenes.
NSD_SYNTHETIC = "nsd_synthetic"


@dataclass(frozen=True)
class DatasetConstants:
    """Canonical per-dataset ROI names, subject IDs and surface space."""

    rois: tuple[str, ...]
    subjects: tuple[str, ...]
    surface_space: str


DATASET_CONSTANTS: dict[str, DatasetConstants] = {
    BOLD_MOMENTS: DatasetConstants(tuple(ROIS), tuple(SUBJECTS), "fsLR32k"),
    NSD: DatasetConstants(tuple(NSD_ROIS), tuple(NSD_SUBJECTS), "fsaverage"),
    NSD_SYNTHETIC: DatasetConstants(
        tuple(NSD_ROIS), tuple(NSD_SUBJECTS), "fsaverage"
    ),
}


def _constants(dataset: str) -> DatasetConstants:
    """Returns the constants for a dataset.

    Args:
        dataset: Dataset name, one of the keys of ``DATASET_CONSTANTS``.

    Returns:
        The matching :class:`DatasetConstants`.

    Raises:
        ValueError: If the dataset is unknown.
    """
    try:
        return DATASET_CONSTANTS[dataset]
    except KeyError:
        raise ValueError(
            f"Unknown dataset {dataset!r}. "
            f"Expected one of {sorted(DATASET_CONSTANTS)}."
        ) from None


def get_rois(
    dataset: str = BOLD_MOMENTS, exclude: Iterable[str] | None = None
) -> list[str]:
    """Returns the canonical ROI names for a dataset.

    Args:
        dataset: Dataset name (default: BOLD Moments, for backwards
            compatibility with the paper's analysis scripts).
        exclude: ROI names to leave out, in canonical order otherwise. An
            alignment pattern is a vector over these ROIs, so excluding one
            changes every pattern, not just that ROI's entry.

    Returns:
        A new list of ROI names.

    Raises:
        ValueError: If a name in ``exclude`` is not an ROI of this dataset,
            which would otherwise silently exclude nothing.
    """
    rois = list(_constants(dataset).rois)
    if not exclude:
        return rois

    excluded = set(exclude)
    unknown = sorted(excluded - set(rois))
    if unknown:
        raise ValueError(
            f"Cannot exclude {unknown} from the {dataset} ROIs: "
            f"no such ROI. Known ROIs are {rois}."
        )
    return [roi for roi in rois if roi not in excluded]


def parse_roi_names(names: str) -> list[str]:
    """Split a comma-separated CLI ROI list into names.

    Args:
        names: Comma-separated ROI names, possibly empty.

    Returns:
        The names, with blanks and surrounding whitespace removed.
    """
    return [name.strip() for name in names.split(",") if name.strip()]


def get_subjects(dataset: str = BOLD_MOMENTS) -> list[str]:
    """Returns the canonical subject IDs for a dataset.

    Args:
        dataset: Dataset name (default: BOLD Moments).

    Returns:
        A new list of subject IDs.
    """
    return list(_constants(dataset).subjects)


def get_surface_space(dataset: str = BOLD_MOMENTS) -> str:
    """Returns the surface space the dataset's fMRI data lives in.

    Args:
        dataset: Dataset name (default: BOLD Moments).

    Returns:
        Either ``"fsLR32k"`` or ``"fsaverage"``.
    """
    return _constants(dataset).surface_space
