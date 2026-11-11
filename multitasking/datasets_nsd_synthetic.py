"""NSD-synthetic as a benchmark dataset.

NSD-synthetic adds one scan session to the eight NSD-core subjects, showing
deliberately non-naturalistic images: noise, Mooney and line-drawing scenes,
contrast and phase-coherence ladders, single words, spiral gratings. Same
scanner, same GLMsingle pipeline, same fsaverage surface, same HCP-MMP parcels.

So this subclasses :class:`~multitasking.datasets_nsd.NSD` rather than
reimplementing the contract: stimulus medium, surface space, subject naming,
atlas, cache format and missing-value imputation are all shared, and only three
things genuinely differ.

*There is no split to impose.* Every subject completed the one session in full
and every image was shown at least twice, so there is no intersection to take
and nothing to drop. All 220 shared stimuli form the test split, and this class
refuses to produce a train split - it exists to be paired with an NSD-core
train split through ``test_split`` in the config.

*The stimuli are a different set under the same numbers.* NSD-core stimuli
1-220 already exist on disk as ``0001.png``..``0220.png``. Sample IDs here are
therefore ``syn0001``..``syn0220``, because the feature caches verify sample
IDs and sample IDs are strings: a shared namespace would let a mis-pointed
stimulus path pair natural images with synthetic betas while every check still
passed.

*The noise ceiling is computed, not published.* NSD ships no NCSNR for
NSD-synthetic in any space.
"""

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from multitasking.datasets_nsd import NSD
from multitasking.fmri_data.noise_ceiling import load_nsd_ncsnr
from multitasking.nsd_synthetic.design import (
    NSD_SYNTHETIC_N_ACHROMATIC,
    NSD_SYNTHETIC_STIMULUS_SET,
    achromatic_stimulus_ids,
    format_sample_id,
    load_trial_counts,
)
from multitasking.nsd_synthetic.prepare_betas import (
    NSD_SYNTHETIC_BETAS_DIRNAME,
    NSD_SYNTHETIC_CACHE_DIRNAME,
)
from multitasking.nsd_synthetic.prepare_stimuli import PROVENANCE_FILENAME

LOGGER = logging.getLogger(__name__)

#: Where the NCSNR comes from. "nsd_synthetic" uses the map computed from this
#: dataset's own repetitions; "nsd_core" reuses NSD-core's, which makes voxel
#: selection identical across the two datasets at the cost of being estimated
#: on natural images.
NcsnrSource = Literal["nsd_synthetic", "nsd_core"]


class NSDSynthetic(NSD):
    """Loads NSD-synthetic stimuli and fMRI responses as a test set."""

    CACHE_DIRNAME: ClassVar[str] = NSD_SYNTHETIC_CACHE_DIRNAME
    STIMULUS_SET: ClassVar[str] = NSD_SYNTHETIC_STIMULUS_SET
    N_STIMULI: ClassVar[int] = NSD_SYNTHETIC_N_ACHROMATIC

    PREPARE_STIMULI_HINT: ClassVar[str] = (
        "Extract the stimuli with "
        "`python -m multitasking.nsd_synthetic.prepare_stimuli`."
    )
    PREPARE_BETAS_HINT: ClassVar[str] = (
        "Build it with `python -m multitasking.nsd_synthetic.prepare_betas "
        "--fmri-path {fmri_path} --output-dir {roi_beta_cache}`."
    )

    def __init__(
        self,
        stimulus_path: str | os.PathLike,
        fmri_path: str | os.PathLike,
        rois: list[str],
        split: Literal["train", "test"],
        subject_ids: str | Sequence[str] = "all",
        surface_space: str = "fsaverage",
        atlas_dir: str | os.PathLike | None = None,
        subsample_stride: int = 6,
        roi_beta_cache: str | os.PathLike | None = None,
        resolution: int | tuple[int, int] | None = None,
        roi_groups: dict[str, list[str]] | None = None,
        ncsnr_source: NcsnrSource = "nsd_synthetic",
        nsd_core_fmri_path: str | os.PathLike | None = None,
    ):
        """Initializes the NSD-synthetic dataset.

        Args:
            stimulus_path: Directory holding the extracted stimuli, i.e.
                containing an ``images/`` subdirectory. Produced by
                ``python -m multitasking.nsd_synthetic.prepare_stimuli``.
            fmri_path: NSD-synthetic root, containing ``nsddata_betas``.
            rois: HCP-MMP parcel names to extract.
            split: Must be "test"; see the class docstring.
            subject_ids: Subjects to score, or "all".
            surface_space: Surface space of the betas.
            atlas_dir: Directory with the ``.annot`` files. Defaults to
                ``<fmri_path>/atlas``, so point it at NSD-core's atlas, which
                is subject- and stimulus-independent.
            subsample_stride: Vertex decimation stride. Must match the dataset
                this one is paired with, or their voxels do not line up.
            roi_beta_cache: Root of the per-ROI beta cache written by
                ``python -m multitasking.nsd_synthetic.prepare_betas``.
            resolution: Resize stimuli to this size.
            roi_groups: Optional ``{group_name: [roi names]}`` to merge.
            ncsnr_source: "nsd_synthetic" for the map computed from this
                dataset's own repetitions, "nsd_core" to reuse NSD-core's.
            nsd_core_fmri_path: NSD-core root. Required only when
                ``ncsnr_source`` is "nsd_core", since ``fmri_path`` then points
                at the wrong tree.

        Raises:
            ValueError: If ``split`` is not "test", if ``ncsnr_source`` is
                unknown, or if the stimulus directory holds a different
                stimulus set.
        """
        if ncsnr_source not in ("nsd_synthetic", "nsd_core"):
            raise ValueError(
                f"Unknown ncsnr_source {ncsnr_source!r}; expected "
                "'nsd_synthetic' or 'nsd_core'."
            )
        self.ncsnr_source: NcsnrSource = ncsnr_source
        self.nsd_core_fmri_path = (
            Path(nsd_core_fmri_path).resolve()
            if nsd_core_fmri_path is not None
            else None
        )
        if ncsnr_source == "nsd_core" and self.nsd_core_fmri_path is None:
            raise ValueError(
                "ncsnr_source='nsd_core' needs nsd_core_fmri_path, since "
                "fmri_path points at the NSD-synthetic tree."
            )

        # Read before super().__init__, which calls _select_stimulus_ids.
        self._design_trial_counts = load_trial_counts()

        super().__init__(
            stimulus_path=stimulus_path,
            fmri_path=fmri_path,
            rois=rois,
            split=split,
            subject_ids=subject_ids,
            stimulus_subject_ids=None,
            min_n_trials=2,
            train_fraction=1.0,
            surface_space=surface_space,
            atlas_dir=atlas_dir,
            subsample_stride=subsample_stride,
            roi_beta_cache=roi_beta_cache,
            resolution=resolution,
            roi_groups=roi_groups,
        )

        self._check_stimulus_provenance()

    @staticmethod
    def format_sample_id(stimulus_id: int) -> str:
        """Formats a stimulus ID as ``synNNNN``.

        Args:
            stimulus_id: The 1-based stimulus ID.

        Returns:
            For example ``"syn0042"``.
        """
        return format_sample_id(stimulus_id)

    def _select_stimulus_ids(self, split: str) -> list[int]:
        """Returns the shared achromatic stimulus IDs, for the test split only.

        Args:
            split: Must be "test".

        Returns:
            ``[1, 2, ..., 220]``.

        Raises:
            ValueError: For any split other than "test".
        """
        if split != "test":
            raise ValueError(
                f"NSD-synthetic has no {split!r} split: it is one session of "
                "220 shared stimuli, meant to be scored against an encoding "
                "model fit elsewhere. Pair it with an NSD-core train split by "
                "putting it under `test_split` in the config, as "
                "configs/benchmark_nsd_to_synthetic.yaml does."
            )

        stimulus_ids = achromatic_stimulus_ids()
        counts = self._design_trial_counts[: len(stimulus_ids)]
        LOGGER.info(
            "NSD-synthetic test split: %d shared achromatic stimuli, %d-%d "
            "repetitions each, %d trials",
            len(stimulus_ids),
            int(counts.min()),
            int(counts.max()),
            int(counts.sum()),
        )
        return stimulus_ids

    def _expected_trial_counts(
        self, subject_id: str, max_reps: int | None = None
    ) -> NDArray[np.uint8]:
        """Returns this split's repetition counts, derived from the design.

        The same for every subject: all eight completed the single session in
        full. A weaker check than NSD-core's, where counts differ by subject
        and two independent derivations are compared - but it still catches a
        preparation step that averaged the wrong trials, because the cache's
        counts come from the averaging and these come from the design file.

        Args:
            subject_id: The subject to look up; unused, see above.
            max_reps: The cap the cache declares on how many repetitions it
                averaged.

        Returns:
            One count per sample, in :attr:`sample_ids` order.
        """
        del subject_id  # Identical for every subject.
        rows = np.asarray(self._stimulus_ids) - 1
        counts = self._design_trial_counts[rows]
        if max_reps is not None:
            counts = np.minimum(counts, max_reps)
        return counts.astype(np.uint8)

    def _check_stimulus_provenance(self) -> None:
        """Checks the stimulus directory holds the stimulus set we expect.

        The sample-ID namespace already makes a collision with NSD-core's
        stimuli impossible, but only once the images are being read. This fails
        at construction instead, and names what is actually there.

        Raises:
            ValueError: If the provenance names a different stimulus set.
        """
        path = self.stimulus_path / PROVENANCE_FILENAME
        if not path.exists():
            LOGGER.warning(
                "No %s in %s, so the stimulus set cannot be verified. %s",
                PROVENANCE_FILENAME,
                self.stimulus_path,
                self.PREPARE_STIMULI_HINT,
            )
            return

        with open(path) as handle:
            provenance = json.load(handle)

        found = provenance.get("stimulus_set")
        if found != self.STIMULUS_SET:
            raise ValueError(
                f"{path} declares stimulus set {found!r}, but this dataset "
                f"scores {self.STIMULUS_SET!r}. The stimulus path points at "
                "the wrong directory."
            )

    def load_voxel_consistency(self, subject_id: str) -> dict[str, np.ndarray]:
        """Loads a subject's per-ROI noise ceiling SNR.

        Note that on the benchmark path this is not what masks the voxels: the
        mask is built from the *train* dataset, which in the intended
        cross-dataset configuration is NSD-core. This is consulted by the
        inter-subject path and by anything normalising by a noise ceiling.

        Args:
            subject_id: The subject whose reliability values to load.

        Returns:
            ``{roi_name: array}`` of per-vertex NCSNR.
        """
        if self.ncsnr_source == "nsd_core":
            assert self.nsd_core_fmri_path is not None  # checked in __init__
            ncsnr = load_nsd_ncsnr(
                self.nsd_core_fmri_path,
                subject_id,
                surface_space=self.surface_space,
            )
        else:
            ncsnr = load_nsd_ncsnr(
                self.fmri_path,
                subject_id,
                surface_space=self.surface_space,
                betas_dirname=NSD_SYNTHETIC_BETAS_DIRNAME,
                missing_hint=(
                    "NSD publishes no noise ceiling for NSD-synthetic; it is "
                    "computed by `python -m "
                    "multitasking.nsd_synthetic.prepare_betas`."
                ),
            )

        # The ROI index arrays address the full-resolution surface, so the
        # untouched NCSNR map is sliced with exactly the same indices as the
        # betas were.
        return {
            roi_name: ncsnr[self.roi_masks.rois[roi_name]]
            for roi_name in self.roi_masks.roi_names
        }
