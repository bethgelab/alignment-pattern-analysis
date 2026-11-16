"""Extracts per-ROI NSD-synthetic betas and noise ceilings into a cache.

NSD-synthetic ships one float32 surface per hemisphere per subject, about
488 MB each, holding one beta per *trial* rather than per image. Turning those
into something a benchmark run can use takes three steps, all done here in a
single pass because reading 7.8 GB twice is the dominant cost:

1. *Z-score each vertex across the session's 744 trials.* NSD-core's prepared
   betas are z-scored per session, and NSD-synthetic is a single session, so
   this is the exact analogue. Everything below depends on it: the noise
   ceiling takes the total variance to be 1, and the dataset fills missing
   cells with 0 as the z-scored expectation.

2. *Estimate NCSNR.* NSD publishes no noise ceiling for NSD-synthetic in any
   space, so it is computed here with NSD's own estimator, pooling the
   within-image variance over all 284 images. Pooling rather than averaging
   per-image variances matters: repetition counts here range from 2 to 10.

3. *Average each image's repetitions and keep only the ROI vertices.* The
   result is roughly 5 MB per subject rather than 976 MB.

Usage:
    python -m multitasking.nsd_synthetic.prepare_betas \
        --fmri-path /path/to/NSD-synthetic/raw \
        --expdesign /path/to/nsdsynthetic_expdesign.mat \
        --output-dir /path/to/cache
"""

import logging
from pathlib import Path
from typing import Any, cast

import click
import numpy as np
from numpy.typing import NDArray

from multitasking.CONSTANTS import NSD_ROIS, NSD_SUBJECTS
from multitasking.fmri_data.noise_ceiling import nsd_ncsnr_path
from multitasking.fmri_data.roi_utils import RoiMasks, build_roi_masks
from multitasking.nsd.prepare_roi_betas import DEFAULT_ATLAS_SUBDIR, cache_path
from multitasking.nsd_synthetic.design import (
    NSD_SYNTHETIC_N_ACHROMATIC,
    NSD_SYNTHETIC_N_IMAGES,
    NSD_SYNTHETIC_N_TRIALS,
    NSD_SYNTHETIC_STIMULUS_SET,
    compute_trial_counts,
    load_masterordering,
    repetitions_by_image,
)

LOGGER = logging.getLogger(__name__)

#: Directory component under the cache root, a sibling of NSD-core's
#: ``nsd_roi_betas``. Kept distinct so that the dataset's superset-cache lookup
#: can never reach from one stimulus set into the other.
NSD_SYNTHETIC_CACHE_DIRNAME = "nsdsynthetic_roi_betas"

#: Beta directory NSD-synthetic publishes under, per subject and space.
NSD_SYNTHETIC_BETAS_DIRNAME = "nsdsyntheticbetas_fithrf_GLMdenoise_RR"

HEMISPHERES = ("lh", "rh")


def betas_path(
    fmri_path: Path, subject_id: str, hemisphere: str, surface_space: str
) -> Path:
    """Returns the path of one hemisphere's published NSD-synthetic betas.

    Args:
        fmri_path: NSD-synthetic root, the directory containing
            ``nsddata_betas``.
        subject_id: For example "subj01".
        hemisphere: "lh" or "rh".
        surface_space: Currently only "fsaverage".

    Returns:
        The ``.mgh`` path.

    Raises:
        NotImplementedError: For a surface space that is not published as a
            surface file.
    """
    if surface_space != "fsaverage":
        raise NotImplementedError(
            f"NSD-synthetic betas are read from fsaverage surfaces, not "
            f"{surface_space!r}. Only nativesurface and fsaverage are "
            "published as surfaces; fsLR32k would have to be resampled first."
        )

    return (
        Path(fmri_path)
        / "nsddata_betas"
        / "ppdata"
        / subject_id
        / "fsaverage"
        / NSD_SYNTHETIC_BETAS_DIRNAME
        / f"{hemisphere}.betas_nsdsynthetic.mgh"
    )


def load_hemisphere_betas(path: Path) -> NDArray[np.float32]:
    """Loads one hemisphere's per-trial betas.

    Args:
        path: The ``.mgh`` file.

    Returns:
        An array of shape ``(n_vertices, 744)``.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If it does not hold one value per stimulus trial.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"NSD-synthetic betas not found at {path}. Download them with "
            "`./tasks/nsd_synthetic/download.sh`."
        )

    # Imported lazily, as elsewhere in the fsaverage path.
    import nibabel
    from nibabel.spatialimages import SpatialImage

    # asanyarray on dataobj rather than get_fdata(), which would upcast to
    # float64 and double the 488 MB. nibabel.load is typed as returning the
    # FileBasedImage base class; an .mgh surface is always a SpatialImage,
    # which is what carries dataobj.
    image = cast(SpatialImage, nibabel.load(path))
    betas = np.asanyarray(image.dataobj, dtype=np.float32)
    betas = betas.squeeze()

    if betas.ndim != 2 or betas.shape[1] != NSD_SYNTHETIC_N_TRIALS:
        raise ValueError(
            f"Expected betas of shape (n_vertices, {NSD_SYNTHETIC_N_TRIALS}) "
            f"in {path}, got {betas.shape}."
        )
    return betas


def zscore_vertices(betas: NDArray[np.float32]) -> tuple[NDArray[np.float32], int]:
    """Z-scores each vertex across the session's trials, in place.

    NSD-core's prepared betas are z-scored per session; NSD-synthetic is one
    session, so this is the same operation. Vertices with no variance - the
    medial wall and anything else outside the acquired volume - are set to zero
    rather than left as NaN, so that they are dropped by the voxel-consistency
    threshold instead of aborting a run.

    Args:
        betas: Per-trial betas of shape ``(n_vertices, n_trials)``.

    Returns:
        ``(betas, n_dead)``, the z-scored array and the number of vertices with
        no variance.
    """
    mean = betas.mean(axis=1, keepdims=True)
    std = betas.std(axis=1, keepdims=True)

    dead = (std == 0) | ~np.isfinite(std) | ~np.isfinite(mean)
    # Avoid dividing by zero; those vertices are overwritten immediately after.
    safe_std = np.where(dead, np.float32(1.0), std)

    betas -= mean
    betas /= safe_std
    betas[dead.squeeze(axis=1)] = 0.0
    betas[~np.isfinite(betas)] = 0.0

    return betas, int(dead.sum())


def compute_ncsnr(
    betas: NDArray[np.float32], repetitions: list[NDArray[np.int64]]
) -> NDArray[np.float32]:
    """Estimates per-vertex noise ceiling SNR from repeated presentations.

    NSD's estimator: with the betas z-scored, total variance is 1, so the
    signal standard deviation follows from the noise standard deviation alone.
    The noise variance is pooled across images rather than averaged per image,
    which is the right generalisation when repetition counts differ - here they
    range from 2 to 10.

    Args:
        betas: Z-scored betas of shape ``(n_vertices, n_trials)``.
        repetitions: Trial indices per image, from
            :func:`~multitasking.nsd_synthetic.design.repetitions_by_image`.

    Returns:
        A per-vertex NCSNR array of shape ``(n_vertices,)``.
    """
    residual_sum = np.zeros(betas.shape[0], dtype=np.float64)
    degrees_of_freedom = 0

    for trials in repetitions:
        if len(trials) < 2:
            continue
        repeats = betas[:, trials]
        residual_sum += np.square(
            repeats - repeats.mean(axis=1, keepdims=True)
        ).sum(axis=1)
        degrees_of_freedom += len(trials) - 1

    noise_variance = residual_sum / max(degrees_of_freedom, 1)
    signal_variance = np.clip(1.0 - noise_variance, 0.0, None)

    with np.errstate(divide="ignore", invalid="ignore"):
        ncsnr = np.sqrt(signal_variance) / np.sqrt(noise_variance)
    ncsnr[~np.isfinite(ncsnr)] = 0.0

    return ncsnr.astype(np.float32)


def average_repetitions(
    betas: NDArray[np.float32],
    repetitions: list[NDArray[np.int64]],
    n_stimuli: int,
) -> NDArray[np.float32]:
    """Averages each image's repeated presentations.

    Args:
        betas: Z-scored betas of shape ``(n_vertices, n_trials)``.
        repetitions: Trial indices per image.
        n_stimuli: How many of the images to keep, from the first.

    Returns:
        An array of shape ``(n_stimuli, n_vertices)``.
    """
    averaged = np.empty((n_stimuli, betas.shape[0]), dtype=np.float32)
    for index in range(n_stimuli):
        averaged[index] = betas[:, repetitions[index]].mean(axis=1)
    return averaged


def extract_subject(
    fmri_path: Path,
    subject_id: str,
    roi_masks: RoiMasks,
    repetitions: list[NDArray[np.int64]],
    surface_space: str,
    n_stimuli: int,
    need_ncsnr: bool,
) -> tuple[NDArray[np.float32], dict[str, NDArray[np.float32]]]:
    """Reads one subject's betas and reduces them to ROI vertices and NCSNR.

    Hemispheres are handled one at a time and released immediately, which keeps
    peak memory at roughly one hemisphere rather than both.

    Args:
        fmri_path: NSD-synthetic root.
        subject_id: For example "subj01".
        roi_masks: Masks whose indices address the concatenated ``[lh, rh]``
            surface.
        repetitions: Trial indices per image.
        surface_space: The surface space.
        n_stimuli: How many images to keep.
        need_ncsnr: Whether to compute the noise ceiling as well.

    Returns:
        ``(betas, ncsnr)`` where ``betas`` has shape
        ``(n_stimuli, n_roi_vertices)`` and ``ncsnr`` maps "ncsnr_lh"/"ncsnr_rh"
        onto whole-hemisphere arrays, empty when ``need_ncsnr`` is False.
    """
    hemisphere_betas: dict[str, NDArray[np.float32]] = {}
    hemisphere_ncsnr: dict[str, NDArray[np.float32]] = {}
    vertex_counts: dict[str, int] = {}

    for hemisphere in HEMISPHERES:
        path = betas_path(fmri_path, subject_id, hemisphere, surface_space)
        LOGGER.info("%s: reading %s", subject_id, path)
        betas = load_hemisphere_betas(path)
        vertex_counts[hemisphere] = betas.shape[0]

        betas, n_dead = zscore_vertices(betas)
        if n_dead:
            LOGGER.info(
                "%s %s: %d of %d vertices have no variance and were zeroed.",
                subject_id,
                hemisphere,
                n_dead,
                betas.shape[0],
            )

        if need_ncsnr:
            hemisphere_ncsnr[f"ncsnr_{hemisphere}"] = compute_ncsnr(
                betas, repetitions
            )

        hemisphere_betas[hemisphere] = average_repetitions(
            betas, repetitions, n_stimuli
        )
        del betas

    # Concatenated ROI indices split back into per-hemisphere ones, exactly as
    # multitasking.nsd.prepare_roi_betas does, so both caches lay their columns
    # out identically.
    n_verts_lh = vertex_counts["lh"]
    all_indices = np.concatenate(
        [roi_masks.rois[roi_name] for roi_name in roi_masks.roi_names]
    )
    lh_selector = all_indices < n_verts_lh
    per_hemi = {
        "lh": all_indices[lh_selector],
        "rh": all_indices[~lh_selector] - n_verts_lh,
    }

    roi_betas = np.empty((n_stimuli, len(all_indices)), dtype=np.float32)
    for hemisphere, hemi_indices in per_hemi.items():
        if len(hemi_indices) == 0:
            continue
        selector = lh_selector if hemisphere == "lh" else ~lh_selector
        roi_betas[:, selector] = hemisphere_betas[hemisphere][:, hemi_indices]

    return roi_betas, hemisphere_ncsnr


@click.command()
@click.option(
    "--fmri-path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="NSD-synthetic root, the directory containing nsddata_betas.",
)
@click.option(
    "--expdesign",
    "expdesign_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to nsdsynthetic_expdesign.mat.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Cache root to write the per-ROI betas into.",
)
@click.option(
    "--atlas-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory with the .annot files. Defaults to <fmri-path>/atlas.",
)
@click.option("--subject", "subjects", multiple=True, help="Subjects (default: all).")
@click.option("--roi", "rois", multiple=True, help="ROI names (default: NSD_ROIS).")
@click.option("--surface-space", default="fsaverage", show_default=True)
@click.option("--subsample-stride", default=6, show_default=True)
@click.option(
    "--max-reps",
    default=None,
    type=int,
    help=(
        "Average at most this many repetitions per image, the earliest ones. "
        "NSD-core stimuli have 2-3 repetitions and NSD-synthetic's have 2-10, "
        "so trial averages differ in noise variance; capping gives a "
        "sensitivity check. Default: use all of them."
    ),
)
@click.option("--overwrite/--no-overwrite", default=False, show_default=True)
def prepare_betas(
    fmri_path: Path,
    expdesign_path: Path,
    output_dir: Path,
    atlas_dir: Path | None,
    subjects: tuple[str, ...],
    rois: tuple[str, ...],
    surface_space: str,
    subsample_stride: int,
    max_reps: int | None,
    overwrite: bool,
) -> None:
    """Writes a per-ROI beta cache and a noise ceiling for each subject."""
    subject_ids = list(subjects) if subjects else list(NSD_SUBJECTS)
    roi_names = list(rois) if rois else list(NSD_ROIS)
    atlas_dir = atlas_dir or (fmri_path / DEFAULT_ATLAS_SUBDIR)

    master_ordering = load_masterordering(expdesign_path)
    design_counts = compute_trial_counts(master_ordering)
    repetitions = repetitions_by_image(master_ordering, max_reps=max_reps)
    n_stimuli = NSD_SYNTHETIC_N_ACHROMATIC

    trial_counts = design_counts[:n_stimuli]
    if max_reps is not None:
        trial_counts = np.minimum(trial_counts, max_reps).astype(np.uint8)
    LOGGER.info(
        "%d of %d images kept (the shared achromatic set), %d trials, "
        "%d-%d repetitions each",
        n_stimuli,
        NSD_SYNTHETIC_N_IMAGES,
        int(trial_counts.sum()),
        int(trial_counts.min()),
        int(trial_counts.max()),
    )

    roi_masks = build_roi_masks(
        roi_names,
        surface_space=surface_space,
        atlas_dir=atlas_dir,
        subsample_stride=subsample_stride,
    )
    roi_masks.extract_rois()
    # extract_rois may reorder or drop names; the cache must record what it used.
    roi_names = list(roi_masks.roi_names)
    n_vertices = sum(roi_masks.voxels_per_roi[roi] for roi in roi_names)
    LOGGER.info(
        "%d ROIs, %d vertices at stride %d",
        len(roi_names),
        n_vertices,
        subsample_stride,
    )

    for subject_id in subject_ids:
        target = cache_path(
            output_dir,
            subject_id,
            roi_names,
            surface_space,
            subsample_stride,
            dirname=NSD_SYNTHETIC_CACHE_DIRNAME,
        )
        ncsnr_target = nsd_ncsnr_path(
            fmri_path, subject_id, surface_space, NSD_SYNTHETIC_BETAS_DIRNAME
        )

        need_betas = overwrite or not target.exists()
        need_ncsnr = overwrite or not ncsnr_target.exists()
        if not need_betas and not need_ncsnr:
            LOGGER.info("%s: cache and noise ceiling already exist", subject_id)
            continue

        LOGGER.info("%s: extracting ...", subject_id)
        betas, ncsnr = extract_subject(
            fmri_path,
            subject_id,
            roi_masks,
            repetitions,
            surface_space,
            n_stimuli,
            need_ncsnr,
        )

        if need_ncsnr:
            ncsnr_target.parent.mkdir(parents=True, exist_ok=True)
            # Widened to Any: numpy's stubs otherwise match **kwargs against
            # savez's allow_pickle parameter.
            ncsnr_arrays: dict[str, Any] = dict(ncsnr)
            np.savez_compressed(ncsnr_target, **ncsnr_arrays)
            inside = np.concatenate(
                [ncsnr["ncsnr_lh"], ncsnr["ncsnr_rh"]]
            )[all_roi_indices(roi_masks)]
            LOGGER.info(
                "%s: wrote %s, median NCSNR in the ROI set %.3f, %.1f%% above "
                "0.2",
                subject_id,
                ncsnr_target,
                float(np.median(inside)),
                100.0 * float((inside >= 0.2).mean()),
            )

        if need_betas:
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                target,
                betas=betas,
                trial_counts=trial_counts,
                stimulus_ids=np.arange(1, n_stimuli + 1),
                roi_names=np.asarray(roi_names),
                roi_vertex_counts=np.asarray(
                    [roi_masks.voxels_per_roi[roi] for roi in roi_names]
                ),
                surface_space=np.asarray(surface_space),
                subsample_stride=np.asarray(subsample_stride),
                normalized=np.asarray(True),
                stimulus_set=np.asarray(NSD_SYNTHETIC_STIMULUS_SET),
                max_reps=np.asarray(-1 if max_reps is None else max_reps),
            )
            LOGGER.info(
                "%s: wrote %s (%.1f MB), %d stimuli",
                subject_id,
                target,
                target.stat().st_size / 1e6,
                n_stimuli,
            )


def all_roi_indices(roi_masks: RoiMasks) -> NDArray[np.int64]:
    """Returns every ROI vertex index, in concatenation order.

    Args:
        roi_masks: The extracted masks.

    Returns:
        The concatenated index array.
    """
    return np.concatenate(
        [roi_masks.rois[roi_name] for roi_name in roi_masks.roi_names]
    )


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(fmt="%(asctime)s %(name)s %(levelname)s %(message)s")
    prepare_betas()
