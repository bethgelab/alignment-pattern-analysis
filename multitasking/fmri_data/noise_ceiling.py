"""Noise ceiling SNR for NSD.

BOLD Moments ships per-split noise ceilings as pickles, loaded by
``multitasking.fmri_data.voxel_consistency``. NSD instead ships a single
per-subject NCSNR map per hemisphere, computed over all of that subject's
stimuli, so it needs its own loader.

Worth stating explicitly, since it affects how the two datasets' voxel
selections compare: NSD's NCSNR is estimated over a subject's full ~10k
stimulus set, whereas BOLD Moments' is estimated over the training split with
three repetitions. The values are comparable in scale and NSD's estimate is
based on more data, but they are not computed identically.
"""

import logging
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


def nsd_ncsnr_path(
    fmri_path: str | Path, subject_id: str, surface_space: str = "fsaverage"
) -> Path:
    """Returns the path of a subject's NCSNR file.

    Args:
        fmri_path: NSD root, the directory containing ``nsddata_betas``.
        subject_id: For example "subj01".
        surface_space: Currently only "fsaverage".

    Returns:
        The ``.npz`` path.

    Raises:
        NotImplementedError: For a surface space whose NCSNR is not prepared.
    """
    if surface_space != "fsaverage":
        raise NotImplementedError(
            f"NSD NCSNR is only available in fsaverage space, not "
            f"{surface_space!r}. For fsLR32k it would have to be recomputed "
            "with nsd_recompute_compare_ncsnr.py from the metrics_benchmark "
            "repository."
        )

    return (
        Path(fmri_path)
        / "nsddata_betas"
        / "ppdata"
        / subject_id
        / "fsaverage"
        / "betas_fithrf_GLMdenoise_RR"
        / "prepared_betas"
        / f"{subject_id}_ncsnr.npz"
    )


def load_nsd_ncsnr(
    fmri_path: str | Path, subject_id: str, surface_space: str = "fsaverage"
) -> np.ndarray:
    """Loads a subject's whole-surface NCSNR.

    The hemispheres are concatenated in the same ``[left, right]`` order the
    betas use, so the result can be sliced with the very same ROI index arrays.

    Args:
        fmri_path: NSD root.
        subject_id: For example "subj01".
        surface_space: Currently only "fsaverage".

    Returns:
        A 1-D array of length ``n_vertices_lh + n_vertices_rh``.

    Raises:
        FileNotFoundError: If the NCSNR file is missing.
        ValueError: If the file does not hold both hemispheres.
    """
    path = nsd_ncsnr_path(fmri_path, subject_id, surface_space)
    if not path.exists():
        raise FileNotFoundError(
            f"NSD noise ceiling not found at {path}. It is produced by "
            "nsd_ncnsr_betas_fsaverage.py from the metrics_benchmark "
            "repository."
        )

    with np.load(path) as data:
        missing = [key for key in ("ncsnr_lh", "ncsnr_rh") if key not in data]
        if missing:
            raise ValueError(f"{path} is missing {missing}; found {list(data)}.")
        return np.concatenate([data["ncsnr_lh"], data["ncsnr_rh"]])
