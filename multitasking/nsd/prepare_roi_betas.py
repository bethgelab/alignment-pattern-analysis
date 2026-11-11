"""Extracts per-ROI NSD betas into a compact per-subject cache.

NSD's prepared betas are monolithic: one ~3.9 GB pickle per subject holding
every vertex of both hemispheres. A benchmark run needs only the vertices
inside the configured ROIs, and it builds a train and a test dataset
separately, so reading the pickles directly would mean ~63 GB of I/O per job
and again for every model evaluated.

This script does that extraction once. The result is roughly 40 MB per
subject, a hundredfold reduction, and it is what every NSD run reads.

Two deliberate choices:

* ``min_n_trials`` is *not* baked in. All 1000 rows are stored together with
  their repetition counts, so one cache serves every threshold.
* Trials are averaged with ``np.nanmean``, which for any stimulus with at
  least one valid repetition is arithmetically identical to imputing missing
  trials with the trial mean and then averaging. Stimuli with no valid
  repetition stay NaN and are excluded by any ``min_n_trials >= 1``.

Usage:
    python -m multitasking.nsd.prepare_roi_betas \
        --fmri-path /path/to/NSD --output-dir path/to/cache
"""

import hashlib
import json
import logging
import pickle
from pathlib import Path

import click
import numpy as np

from multitasking.CONSTANTS import NSD_ROIS, NSD_SUBJECTS
from multitasking.fmri_data.roi_utils import RoiMasks, build_roi_masks
from multitasking.nsd.trial_counts import NSD_SHARED1000_SIZE

LOGGER = logging.getLogger(__name__)

DEFAULT_ATLAS_SUBDIR = "atlas"


def roi_set_hash(
    roi_names: list[str], surface_space: str, subsample_stride: int
) -> str:
    """Returns a short, stable digest identifying a cached ROI selection.

    Args:
        roi_names: ROI names, in the order they are concatenated.
        surface_space: The surface space the ROIs were extracted in.
        subsample_stride: Vertex decimation stride.

    Returns:
        A 12-character hex digest.
    """
    payload = json.dumps(
        {
            "roi_names": list(roi_names),
            "surface_space": surface_space,
            "subsample_stride": subsample_stride,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def cache_path(
    output_dir: Path,
    subject_id: str,
    roi_names: list[str],
    surface_space: str,
    subsample_stride: int,
) -> Path:
    """Returns where a subject's per-ROI cache lives.

    Args:
        output_dir: Cache root.
        subject_id: For example "subj01".
        roi_names: ROI names, in concatenation order.
        surface_space: The surface space.
        subsample_stride: Vertex decimation stride.

    Returns:
        The ``.npz`` path.
    """
    digest = roi_set_hash(roi_names, surface_space, subsample_stride)
    return (
        output_dir
        / "nsd_roi_betas"
        / surface_space
        / f"stride{subsample_stride}"
        / digest
        / f"{subject_id}.npz"
    )


def prepared_betas_path(
    fmri_path: Path, subject_id: str, surface_space: str, normalized: bool
) -> Path:
    """Returns the path of a subject's prepared betas.

    Args:
        fmri_path: NSD root, the directory containing ``nsddata_betas``.
        subject_id: For example "subj01".
        surface_space: Currently only "fsaverage".
        normalized: Whether to use the session-z-scored betas.

    Returns:
        The pickle path.

    Raises:
        NotImplementedError: For a surface space whose betas are not prepared.
    """
    if surface_space != "fsaverage":
        raise NotImplementedError(
            f"NSD betas are only prepared in fsaverage space, not "
            f"{surface_space!r}. Converting them would mean running "
            "nsd_convert_fsaverage_to_fslr32k.py and "
            "nsd_organize_betas_fslr32k.py from the metrics_benchmark "
            "repository; the output directories are currently empty."
        )

    suffix = "normalized" if normalized else "unnormalized"
    return (
        Path(fmri_path)
        / "nsddata_betas"
        / "ppdata"
        / subject_id
        / "fsaverage"
        / "betas_fithrf_GLMdenoise_RR"
        / "prepared_betas"
        / f"{subject_id}_organized_betas_task-train_{suffix}_fixed.pkl"
    )


def extract_subject_roi_betas(
    betas_path: Path, roi_masks: RoiMasks
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Averages a subject's betas over trials and keeps only the ROI vertices.

    Hemispheres are sliced separately rather than concatenated, which halves
    peak memory on arrays of this size.

    Args:
        betas_path: The subject's prepared-betas pickle.
        roi_masks: Masks whose indices address the concatenated ``[lh, rh]``
            surface.

    Returns:
        ``(betas, trial_counts, stimulus_ids)``, where ``betas`` has shape
        ``(n_stimuli, n_roi_vertices)``.

    Raises:
        FileNotFoundError: If the betas are missing.
        ValueError: If the file's shape is not what the loader expects.
    """
    if not betas_path.exists():
        raise FileNotFoundError(f"Prepared NSD betas not found at {betas_path}")

    with open(betas_path, "rb") as handle:
        data = pickle.load(handle)

    stimulus_ids = np.asarray(data["stimulus_indices"])
    n_verts_lh = data["lh"].shape[-1]

    if data["lh"].shape[0] != NSD_SHARED1000_SIZE:
        raise ValueError(
            f"Expected {NSD_SHARED1000_SIZE} stimuli in {betas_path}, got "
            f"{data['lh'].shape[0]}."
        )
    if data["rh"].shape[-1] != n_verts_lh:
        raise ValueError(
            f"Hemispheres differ in vertex count in {betas_path}: "
            f"{n_verts_lh} vs {data['rh'].shape[-1]}."
        )

    # Repetition counts, read off the NaN pattern of a single vertex. These let
    # the dataset re-check the counts derived from the experiment design.
    trial_counts = (~np.isnan(data["lh"][:, :, 0])).sum(axis=1).astype(np.uint8)

    # Concatenated indices split back into per-hemisphere ones, so each
    # hemisphere can be averaged and released before touching the other.
    all_indices = np.concatenate(
        [roi_masks.rois[roi_name] for roi_name in roi_masks.roi_names]
    )
    lh_selector = all_indices < n_verts_lh
    per_hemi = {
        "lh": all_indices[lh_selector],
        "rh": all_indices[~lh_selector] - n_verts_lh,
    }

    betas = np.empty((len(stimulus_ids), len(all_indices)), dtype=np.float32)
    for hemi, hemi_indices in per_hemi.items():
        if len(hemi_indices) == 0:
            continue
        selector = lh_selector if hemi == "lh" else ~lh_selector
        with np.errstate(invalid="ignore"):
            # All-NaN stimuli warn here; they are dropped later by min_n_trials.
            betas[:, selector] = np.nanmean(
                data[hemi][:, :, hemi_indices], axis=1
            ).astype(np.float32)
        del data[hemi]

    return betas, trial_counts, stimulus_ids


@click.command()
@click.option(
    "--fmri-path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="NSD root, the directory containing nsddata_betas.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Cache root to write into.",
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
@click.option("--normalized/--unnormalized", default=True, show_default=True)
@click.option("--overwrite/--no-overwrite", default=False, show_default=True)
def prepare_roi_betas(
    fmri_path: Path,
    output_dir: Path,
    atlas_dir: Path | None,
    subjects: tuple[str, ...],
    rois: tuple[str, ...],
    surface_space: str,
    subsample_stride: int,
    normalized: bool,
    overwrite: bool,
) -> None:
    """Writes a compact per-ROI beta cache for each subject."""
    subject_ids = list(subjects) if subjects else list(NSD_SUBJECTS)
    roi_names = list(rois) if rois else list(NSD_ROIS)
    atlas_dir = atlas_dir or (fmri_path / DEFAULT_ATLAS_SUBDIR)

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
        "%d ROIs, %d vertices at stride %d", len(roi_names), n_vertices, subsample_stride
    )

    for subject_id in subject_ids:
        target = cache_path(
            output_dir, subject_id, roi_names, surface_space, subsample_stride
        )
        if target.exists() and not overwrite:
            LOGGER.info("%s: cache already exists at %s", subject_id, target)
            continue

        LOGGER.info("%s: extracting ROI betas ...", subject_id)
        betas, trial_counts, stimulus_ids = extract_subject_roi_betas(
            prepared_betas_path(fmri_path, subject_id, surface_space, normalized),
            roi_masks,
        )

        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            betas=betas,
            trial_counts=trial_counts,
            stimulus_ids=stimulus_ids,
            roi_names=np.asarray(roi_names),
            roi_vertex_counts=np.asarray(
                [roi_masks.voxels_per_roi[roi] for roi in roi_names]
            ),
            surface_space=np.asarray(surface_space),
            subsample_stride=np.asarray(subsample_stride),
            normalized=np.asarray(normalized),
        )
        LOGGER.info(
            "%s: wrote %s (%.1f MB), %d stimuli with >=2 trials",
            subject_id,
            target,
            target.stat().st_size / 1e6,
            int((trial_counts >= 2).sum()),
        )


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(fmt="%(asctime)s %(name)s %(levelname)s %(message)s")
    prepare_roi_betas()
