"""Shared contract for the stimulus + fMRI datasets the benchmark can evaluate.

This module defines :class:`FmriStimulusDataset`, the interface
``multitasking.benchmark`` and ``multitasking.feature_extraction`` rely on. It
is deliberately a *contract* rather than a place to share implementation: the
datasets differ in stimulus medium (video vs. still image), surface space,
subject naming and file layout, so there is very little they could share
without one dataset's quirks leaking into the other.

The single most important part of the contract is ``sample_ids``. Model
features are extracted once per split and reused for every subject, so the
stimulus ordering must be fixed before extraction and must not depend on which
subject is being scored. Every implementation must guarantee that row ``i`` of
each array returned by :meth:`FmriStimulusDataset.load_fmri_data` corresponds
to ``sample_ids[i]``.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Literal

import numpy as np
import torch

from multitasking.fmri_data.roi_utils import RoiMasks


class FmriStimulusDataset(ABC):
    """A set of stimuli paired with fMRI responses from one or more subjects.

    Attributes:
        split: Which split this instance serves, "train" or "test".
        subject_ids: The subjects whose fMRI data this instance can load.
        rois: The requested region-of-interest names.
        roi_masks: Masks translating whole-brain data into per-ROI data.
    """

    split: Literal["train", "test"]
    subject_ids: list[str]
    rois: list[str]
    roi_masks: RoiMasks

    @property
    @abstractmethod
    def sample_ids(self) -> list[str]:
        """Returns the stimulus IDs of this split, in loading order.

        This is the single source of truth for sample ordering: stimuli, fMRI
        rows and cached model features all follow this list. It must be
        deterministic and independent of which subjects are being scored.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of samples in this split."""

    @abstractmethod
    def load_videos(
        self, batch_size: int = 1
    ) -> Iterator[tuple[torch.Tensor, list[str]]]:
        """Yields batches of stimuli together with their sample IDs.

        Args:
            batch_size: Number of stimuli per batch.

        Yields:
            Tuples of ``(stimuli, ids)``, where ``stimuli`` is a uint8 tensor of
            shape ``(B, C, T, H, W)``. ``T`` is 1 for still-image datasets. The
            concatenated ``ids`` must equal :attr:`sample_ids`.
        """

    @abstractmethod
    def load_fmri_data(self) -> dict[str, dict[str, np.ndarray]]:
        """Loads this split's fMRI data for every subject, per ROI.

        Returns:
            ``{subject_id: {roi_name: array}}``, where each array's first axis
            runs over :attr:`sample_ids` in order.
        """

    @abstractmethod
    def load_voxel_consistency(self, subject_id: str) -> dict[str, np.ndarray]:
        """Loads a subject's per-ROI voxel reliability (noise ceiling SNR).

        Args:
            subject_id: The subject whose reliability values to load.

        Returns:
            ``{roi_name: array}`` of per-voxel values, matching the voxel
            ordering of :meth:`load_fmri_data`.
        """

    def load_stimuli(
        self, batch_size: int = 1
    ) -> Iterator[tuple[torch.Tensor, list[str]]]:
        """Yields batches of stimuli; medium-neutral alias of :meth:`load_videos`.

        Args:
            batch_size: Number of stimuli per batch.

        Returns:
            The iterator returned by :meth:`load_videos`.
        """
        return self.load_videos(batch_size=batch_size)
