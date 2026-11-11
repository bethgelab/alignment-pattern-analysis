"""The Natural Scenes Dataset as a benchmark dataset.

NSD pairs still images with fMRI responses from eight subjects on the
fsaverage surface. Two things differ from BOLD Moments in ways that shape this
class:

*Subjects saw different stimuli.* Sessions were not all completed, so a
stimulus one subject saw three times another may not have seen at all. The
usable stimulus set is therefore the intersection over a fixed set of
subjects, and it is deliberately **not** derived from whichever subjects are
being scored - see ``stimulus_subject_ids`` below.

*NSD ships no train/test split.* One is imposed deterministically on the
stimulus ID, before the intersection, so that a stimulus can never change
split when the subject set changes.
"""

import logging
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from multitasking.CONSTANTS import NSD_SUBJECTS
from multitasking.datasets_base import FmriStimulusDataset
from multitasking.fmri_data.noise_ceiling import load_nsd_ncsnr
from multitasking.fmri_data.roi_utils import build_roi_masks
from multitasking.nsd.prepare_roi_betas import (
    NSD_CACHE_DIRNAME,
    NSD_STIMULUS_SET,
    cache_path,
)
from multitasking.nsd.splits import format_sample_id, split_sample_ids
from multitasking.nsd.trial_counts import (
    NSD_SHARED1000_SIZE,
    load_shared1000_trial_counts,
    valid_stimulus_ids,
)

LOGGER = logging.getLogger(__name__)

SCRATCH = Path(os.environ.get("SCRATCH", "/scratch"))

STIMULUS_IMAGES_DIRNAME = "images"


class NSD(FmriStimulusDataset):
    """Loads NSD stimuli and fMRI responses for one split.

    The handful of class variables below are the seams a parallel stimulus set
    varies. :class:`multitasking.datasets_nsd_synthetic.NSDSynthetic` shares
    this class's surface space, subjects, atlas, cache format and imputation,
    and differs only in which stimuli exist and where their artifacts live.
    """

    #: Directory component under the beta-cache root. Must be used both when
    #: building the exact path and when globbing for a superset cache, or a
    #: subclass's superset lookup would reach into this one's caches.
    CACHE_DIRNAME: ClassVar[str] = NSD_CACHE_DIRNAME

    #: Stimulus set a loaded cache must declare.
    STIMULUS_SET: ClassVar[str] = NSD_STIMULUS_SET

    #: Rows a cache for this stimulus set must hold.
    N_STIMULI: ClassVar[int] = NSD_SHARED1000_SIZE

    #: Told to the caller when a stimulus image is missing.
    PREPARE_STIMULI_HINT: ClassVar[str] = (
        "Reconstruct the stimuli with "
        "`python -m multitasking.nsd.prepare_stimuli`."
    )

    #: Told to the caller when a beta cache is missing. Formatted with
    #: ``fmri_path`` and ``roi_beta_cache``.
    PREPARE_BETAS_HINT: ClassVar[str] = (
        "Build it with `python -m multitasking.nsd.prepare_roi_betas "
        "--fmri-path {fmri_path} --output-dir {roi_beta_cache}`."
    )

    def __init__(
        self,
        stimulus_path: str | os.PathLike,
        fmri_path: str | os.PathLike,
        rois: list[str],
        split: Literal["train", "test"],
        subject_ids: str | Sequence[str] = "all",
        stimulus_subject_ids: Sequence[str] | None = None,
        min_n_trials: int = 2,
        train_fraction: float = 0.8,
        surface_space: str = "fsaverage",
        atlas_dir: str | os.PathLike | None = None,
        subsample_stride: int = 6,
        roi_beta_cache: str | os.PathLike | None = None,
        resolution: int | tuple[int, int] | None = None,
        roi_groups: dict[str, list[str]] | None = None,
    ):
        """Initializes the NSD dataset.

        Args:
            stimulus_path: Directory holding the reconstructed stimuli, i.e.
                containing an ``images/`` subdirectory. Produced by
                ``python -m multitasking.nsd.prepare_stimuli``.
            fmri_path: NSD root, the directory containing ``nsddata_betas``.
            rois: HCP-MMP parcel names to extract.
            split: "train" or "test".
            subject_ids: Subjects to score, or "all".
            stimulus_subject_ids: Subjects whose shared stimuli define the
                usable stimulus set. Defaults to all eight, and deliberately
                does *not* follow ``subject_ids``: the stimulus set must be
                identical across a full benchmark run and the pairwise
                inter-subject runs it is compared against, or model-brain and
                brain-brain alignment would be computed on different stimuli.
            min_n_trials: Minimum repetitions a subject must have for a
                stimulus to be usable.
            train_fraction: Fraction of the shared-1000 assigned to train.
            surface_space: Surface space of the betas.
            atlas_dir: Directory with the ``.annot`` files. Defaults to
                ``<fmri_path>/atlas``.
            subsample_stride: Vertex decimation stride.
            roi_beta_cache: Root of the per-ROI beta cache written by
                ``python -m multitasking.nsd.prepare_roi_betas``. Defaults to
                ``$SCRATCH/nsd_cache``.
            resolution: Resize stimuli to this size, as for BOLD Moments.
            roi_groups: Optional ``{group_name: [roi names]}`` to merge.

        Raises:
            ValueError: If a scored subject is outside ``stimulus_subject_ids``,
                if ``resolution`` is malformed, or if the split ends up empty.
        """
        self.stimulus_path = Path(stimulus_path).resolve()
        self.fmri_path = Path(fmri_path).resolve()
        self.split = split
        self.rois = list(rois)
        self.min_n_trials = min_n_trials
        self.train_fraction = train_fraction
        self.surface_space = surface_space
        self.subsample_stride = subsample_stride
        self.roi_beta_cache = (
            Path(roi_beta_cache) if roi_beta_cache is not None
            else SCRATCH / "nsd_cache"
        )

        if subject_ids == "all":
            self.subject_ids = list(NSD_SUBJECTS)
        elif isinstance(subject_ids, str):
            self.subject_ids = [subject_ids]
        else:
            self.subject_ids = list(subject_ids)

        self.stimulus_subject_ids = (
            list(stimulus_subject_ids)
            if stimulus_subject_ids is not None
            else list(NSD_SUBJECTS)
        )

        outside = set(self.subject_ids) - set(self.stimulus_subject_ids)
        if outside:
            raise ValueError(
                f"Subjects {sorted(outside)} are scored but are not in "
                f"stimulus_subject_ids {self.stimulus_subject_ids}. The stimulus "
                "set would not account for the stimuli they are missing."
            )

        self.roi_masks = build_roi_masks(
            self.rois,
            roi_groups,
            surface_space=surface_space,
            atlas_dir=atlas_dir or (self.fmri_path / "atlas"),
            subsample_stride=subsample_stride,
        )
        self.roi_masks.extract_rois()

        self._stimulus_ids = self._select_stimulus_ids(split)
        self._sample_ids = [
            self.format_sample_id(stimulus_id)
            for stimulus_id in self._stimulus_ids
        ]

        if isinstance(resolution, int):
            self.resolution: tuple[int, int] | None = (resolution, resolution)
        elif isinstance(resolution, tuple):
            if len(resolution) != 2 or resolution[0] != resolution[1]:
                raise ValueError(f"Invalid resolution: {resolution}")
            self.resolution = resolution
        else:
            self.resolution = None

    @staticmethod
    def format_sample_id(stimulus_id: int) -> str:
        """Formats a stimulus ID the way this dataset keys its samples.

        Args:
            stimulus_id: The 1-based stimulus ID.

        Returns:
            The sample ID string.
        """
        return format_sample_id(stimulus_id)

    def _select_stimulus_ids(self, split: str) -> list[int]:
        """Returns this split's stimulus IDs, and fixes the sample ordering.

        Splits the shared-1000 deterministically on the stimulus ID and only
        then intersects with what the stimulus subjects actually saw, so a
        stimulus can never change split when the subject set changes.

        Args:
            split: "train" or "test".

        Returns:
            Sorted, ascending stimulus IDs.

        Raises:
            ValueError: If the split ends up empty.
        """
        self._trial_counts = load_shared1000_trial_counts()
        usable = valid_stimulus_ids(
            self.stimulus_subject_ids, self.min_n_trials, self._trial_counts
        )
        stimulus_ids = split_sample_ids(split, usable, self.train_fraction)
        if not stimulus_ids:
            raise ValueError(
                f"The {split} split is empty for subjects "
                f"{self.stimulus_subject_ids} at "
                f"min_n_trials={self.min_n_trials}."
            )

        LOGGER.info(
            "NSD %s split: %d of %d shared stimuli (%d subjects define the "
            "stimulus set, min_n_trials=%d, train_fraction=%.2f)",
            split,
            len(stimulus_ids),
            len(usable),
            len(self.stimulus_subject_ids),
            self.min_n_trials,
            self.train_fraction,
        )
        return stimulus_ids

    def _expected_trial_counts(
        self, subject_id: str, max_reps: int | None = None
    ) -> NDArray[np.uint8]:
        """Returns this split's repetition counts, derived from the design.

        Independent of the betas, which is the point: comparing it against the
        counts the cache recorded validates the derivation that fixed
        ``sample_ids`` against the data actually loaded.

        Args:
            subject_id: The subject to look up.
            max_reps: A cap the cache declares on how many repetitions it
                averaged. NSD-core caches never set one - they average whatever
                a subject saw - so this is ignored here and honoured by
                subclasses whose preparation step offers the option.

        Returns:
            One count per sample, in :attr:`sample_ids` order.
        """
        del max_reps  # NSD-core averages every available repetition.
        rows = np.asarray(self._stimulus_ids) - 1
        return self._trial_counts[subject_id][rows]

    def __len__(self) -> int:
        """Returns the number of samples in this split."""
        return len(self._sample_ids)

    @property
    def sample_ids(self) -> list[str]:
        """Returns the stimulus IDs of this split, in loading order."""
        return self._sample_ids

    @property
    def stimulus_ids(self) -> list[int]:
        """Returns this split's stimulus IDs as integers in ``1..N_STIMULI``."""
        return self._stimulus_ids

    def _image_path(self, sample_id: str) -> Path:
        """Returns the image file for a sample ID."""
        return self.stimulus_path / STIMULUS_IMAGES_DIRNAME / f"{sample_id}.png"

    def _load_image(self, path: Path) -> torch.Tensor:
        """Loads one stimulus as a (C, 1, H, W) uint8 tensor.

        The singleton time axis keeps still images compatible with the
        ``(B, C, T, H, W)`` contract the feature extractor expects.

        Args:
            path: The image file.

        Returns:
            The image tensor.

        Raises:
            FileNotFoundError: If the stimulus has not been reconstructed.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Stimulus not found at {path}. {self.PREPARE_STIMULI_HINT}"
            )

        with Image.open(path) as source:
            image = source.convert("RGB")
        if self.resolution is not None:
            image = image.resize(self.resolution, Image.Resampling.BILINEAR)

        # np.array copies; torch warns on tensors backed by read-only buffers.
        array = np.array(image, dtype=np.uint8)
        return torch.from_numpy(array).movedim(-1, 0).unsqueeze(1)

    def load_videos(
        self, batch_size: int = 1
    ) -> Iterator[tuple[torch.Tensor, list[str]]]:
        """Yields batches of stimuli with their sample IDs, in sample_ids order.

        Args:
            batch_size: Number of stimuli per batch.

        Yields:
            ``(stimuli, ids)`` with stimuli of shape ``(B, C, 1, H, W)``.
        """
        batch: list[torch.Tensor] = []
        ids: list[str] = []
        for sample_id in self._sample_ids:
            batch.append(self._load_image(self._image_path(sample_id)))
            ids.append(sample_id)
            if len(batch) == batch_size:
                yield torch.stack(batch), ids
                batch, ids = [], []
        if batch:
            yield torch.stack(batch), ids

    def _cache_path(self, subject_id: str) -> Path:
        """Returns a subject's per-ROI beta cache path.

        Prefers a cache built for exactly this ROI set, and otherwise accepts
        one built for a superset, so that a cache prepared once with the full
        ROI list serves every subset without a ~30 minute rebuild.

        Args:
            subject_id: The subject to look up.

        Returns:
            The path of a usable cache, or of the exact-match cache when none
            exists yet, so that the caller's error message names it.
        """
        exact = cache_path(
            self.roi_beta_cache,
            subject_id,
            list(self.roi_masks.roi_names),
            self.surface_space,
            self.subsample_stride,
            dirname=self.CACHE_DIRNAME,
        )
        if exact.exists():
            return exact

        needed = set(self.roi_masks.roi_names)
        search_root = (
            self.roi_beta_cache
            / self.CACHE_DIRNAME
            / self.surface_space
            / f"stride{self.subsample_stride}"
        )
        for candidate in sorted(search_root.glob(f"*/{subject_id}.npz")):
            with np.load(candidate, allow_pickle=False) as cached:
                cached_rois = {str(name) for name in cached["roi_names"]}
            if needed <= cached_rois:
                LOGGER.info(
                    "Using ROI beta cache %s, which covers the %d requested ROIs.",
                    candidate,
                    len(needed),
                )
                return candidate

        return exact

    def _load_subject_betas(self, subject_id: str) -> NDArray[np.float32]:
        """Loads a subject's ROI betas for this split, in sample_ids order.

        Also re-checks the repetition counts recorded in the cache against the
        counts derived from the experiment design, so that the derivation which
        fixed ``sample_ids`` is validated against the data actually loaded.

        Args:
            subject_id: The subject to load.

        Returns:
            An array of shape ``(n_samples, n_roi_vertices)``.

        Raises:
            FileNotFoundError: If the cache is missing.
            ValueError: If the cache disagrees with this dataset's ROI set, or
                if a selected stimulus has too few repetitions.
        """
        path = self._cache_path(subject_id)
        if not path.exists():
            raise FileNotFoundError(
                f"ROI beta cache not found at {path}. "
                + self.PREPARE_BETAS_HINT.format(
                    fmri_path=self.fmri_path,
                    roi_beta_cache=self.roi_beta_cache,
                )
            )

        with np.load(path, allow_pickle=False) as cached:
            betas = cached["betas"]
            trial_counts = cached["trial_counts"]
            stimulus_ids = cached["stimulus_ids"]
            cached_rois = [str(name) for name in cached["roi_names"]]
            cached_counts = cached["roi_vertex_counts"]
            # Caches written before the tag existed all hold NSD-core stimuli.
            cached_stimulus_set = (
                str(cached["stimulus_set"])
                if "stimulus_set" in cached
                else NSD_STIMULUS_SET
            )
            # -1 and absence both mean "every available repetition".
            cached_max_reps = (
                int(cached["max_reps"]) if "max_reps" in cached else -1
            )

        missing_rois = [
            roi for roi in self.roi_masks.roi_names if roi not in cached_rois
        ]
        if missing_rois:
            raise ValueError(
                f"Cached betas at {path} were built for ROIs {cached_rois} and "
                f"do not cover {missing_rois}."
            )

        # The cache may hold more ROIs than this run needs; take the requested
        # ones, in this dataset's ROI order, so that the columns line up with
        # roi_masks.rois_to_indices.
        cached_bounds = {}
        start = 0
        for roi_name, count in zip(cached_rois, cached_counts, strict=True):
            cached_bounds[roi_name] = (start, start + int(count))
            start += int(count)

        for roi_name in self.roi_masks.roi_names:
            cached_width = (
                cached_bounds[roi_name][1] - cached_bounds[roi_name][0]
            )
            expected_width = self.roi_masks.voxels_per_roi[roi_name]
            if cached_width != expected_width:
                raise ValueError(
                    f"Cached betas at {path} have {cached_width} vertices for "
                    f"{roi_name}, but the atlas gives {expected_width}. The "
                    "cache was built with a different atlas or stride."
                )

        betas = np.concatenate(
            [
                betas[:, slice(*cached_bounds[roi_name])]
                for roi_name in self.roi_masks.roi_names
            ],
            axis=1,
        )

        # Identity first. The superset lookup above can land on a cache built
        # for a different stimulus set entirely - one that is internally
        # well-ordered and would pass every check below while holding the wrong
        # brain responses.
        if cached_stimulus_set != self.STIMULUS_SET:
            raise ValueError(
                f"Cached betas at {path} hold the {cached_stimulus_set!r} "
                f"stimulus set, but this dataset scores {self.STIMULUS_SET!r}."
            )
        if len(stimulus_ids) != self.N_STIMULI:
            raise ValueError(
                f"Cached betas at {path} hold {len(stimulus_ids)} stimuli, but "
                f"{self.STIMULUS_SET!r} has {self.N_STIMULI}."
            )

        expected_ids = np.arange(1, len(stimulus_ids) + 1)
        if not np.array_equal(stimulus_ids, expected_ids):
            raise ValueError(
                f"Cached betas at {path} are not in contiguous "
                f"1..{self.N_STIMULI} stimulus order."
            )

        # stimulus id k lives at row k - 1.
        rows = np.asarray(self._stimulus_ids) - 1

        design_counts = self._expected_trial_counts(
            subject_id, max_reps=None if cached_max_reps < 0 else cached_max_reps
        )
        actual_counts = trial_counts[rows]
        if not np.array_equal(design_counts, actual_counts):
            n_differing = int((design_counts != actual_counts).sum())
            raise ValueError(
                f"{subject_id}: repetition counts in {path} disagree with the "
                f"counts derived from the experiment design for {n_differing} "
                "of this split's stimuli. The stimulus selection that fixed "
                "sample_ids cannot be trusted for this subject."
            )

        if (actual_counts < self.min_n_trials).any():
            raise ValueError(
                f"{subject_id}: some selected stimuli have fewer than "
                f"{self.min_n_trials} repetitions."
            )

        selected = betas[rows]
        return self._impute_missing_vertices(selected, subject_id)

    def _impute_missing_vertices(
        self, betas: NDArray[np.float32], subject_id: str
    ) -> NDArray[np.float32]:
        """Fills vertices that have no valid repetition for a given stimulus.

        A handful of (stimulus, vertex) cells have every repetition missing -
        for the ROIs in the default config, 0.013% of subj06's values and nine
        vertices of subj08's, with every other subject clean. The betas are
        z-scored per session, so these are filled with zero, the vertex's
        expected value, which is also what the reference implementation's
        ``impute_nans`` does once trial averaging has been applied.

        Args:
            betas: Selected betas of shape ``(n_samples, n_vertices)``.
            subject_id: The subject, for logging.

        Returns:
            The array with missing cells filled.

        Raises:
            ValueError: If a vertex has no valid value for any stimulus, which
                would leave a constant feature rather than a gap.
        """
        missing = np.isnan(betas)
        if not missing.any():
            return betas

        empty_vertices = np.flatnonzero(missing.all(axis=0))
        if empty_vertices.size:
            raise ValueError(
                f"{subject_id}: {empty_vertices.size} vertices have no valid "
                "value for any stimulus in this split; filling them would "
                "create constant features. Affected vertex indices (first "
                f"ten): {empty_vertices[:10].tolist()}"
            )

        LOGGER.warning(
            "%s: filling %d missing values (%.4f%% of cells, %d vertices) with "
            "zero, the per-session z-scored expectation.",
            subject_id,
            int(missing.sum()),
            100.0 * missing.sum() / missing.size,
            int(np.unique(np.flatnonzero(missing.any(axis=0))).size),
        )
        filled = betas.copy()
        filled[missing] = 0.0
        return filled

    def _reformat_data_roi_wise(
        self, betas: NDArray[np.float32]
    ) -> dict[str, NDArray[np.float32]]:
        """Splits concatenated ROI betas back into one array per ROI.

        Args:
            betas: Array of shape ``(n_samples, n_roi_vertices)``.

        Returns:
            ``{roi_name: array}``.
        """
        return {
            roi_name: betas[..., slice(*self.roi_masks.rois_to_indices[roi_name])]
            for roi_name in self.roi_masks.roi_names
        }

    def load_fmri_data(self) -> dict[str, dict[str, Any]]:
        """Loads this split's fMRI data for every scored subject, per ROI.

        Returns:
            ``{subject_id: {roi_name: array}}``, each array's first axis
            running over :attr:`sample_ids` in order.
        """
        return {
            subject_id: self._reformat_data_roi_wise(
                self._load_subject_betas(subject_id)
            )
            for subject_id in self.subject_ids
        }

    def load_voxel_consistency(self, subject_id: str) -> dict[str, np.ndarray]:
        """Loads a subject's per-ROI noise ceiling SNR.

        Args:
            subject_id: The subject whose reliability values to load.

        Returns:
            ``{roi_name: array}`` of per-vertex NCSNR.
        """
        ncsnr = load_nsd_ncsnr(
            self.fmri_path, subject_id, surface_space=self.surface_space
        )
        # The ROI index arrays address the full-resolution surface, so the
        # untouched NCSNR map is sliced with exactly the same indices as the
        # betas were.
        return {
            roi_name: ncsnr[self.roi_masks.rois[roi_name]]
            for roi_name in self.roi_masks.roi_names
        }
