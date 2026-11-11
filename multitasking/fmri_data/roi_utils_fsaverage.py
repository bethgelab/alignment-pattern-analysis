"""ROI masks for data on the fsaverage surface.

The BOLD Moments pipeline works in fsLR32k and reads the HCP-MMP parcellation
from ``hcp_utils``. NSD's betas are released on the fsaverage surface instead,
where the same parcellation is distributed as a pair of FreeSurfer ``.annot``
files. This module provides a drop-in :class:`RoiMasks` subclass for that case,
so both datasets expose ROIs under the same names.
"""

import logging
from pathlib import Path
from typing import ClassVar

import nibabel as nib
import numpy as np

from multitasking.fmri_data.roi_utils import RoiMasks

LOGGER = logging.getLogger(__name__)

# fsaverage has roughly six times as many vertices per hemisphere as fsLR32k,
# so ROI data is usually decimated to keep the two datasets comparable in size.
DEFAULT_SUBSAMPLE_STRIDE = 6


class RoiMasksFSAverage(RoiMasks):
    """ROI masks read from the fsaverage HCP-MMP1 FreeSurfer annotation.

    Exposes exactly the attributes :class:`RoiMasks` does - ``roi_names``,
    ``rois``, ``rois_to_indices``, ``voxels_per_roi`` and ``hcp_roi_mask`` - so
    that everything downstream, including the inherited
    :meth:`RoiMasks.merge_roi_groups` and :meth:`RoiMasks.map_rois_to_voxels`,
    works unchanged.

    Vertex indices address the concatenated ``[left, right]`` surface, matching
    the order in which the hemispheres' betas are concatenated. They remain
    indices into the full-resolution surface even when subsampled, which is what
    lets full-resolution per-vertex data (such as NSD's noise ceiling SNR) be
    sliced with the very same index arrays.
    """

    surface_space: ClassVar[str] = "fsaverage"

    def __init__(
        self,
        roi_names: list[str],
        roi_groups: dict[str, list[str]] | None = None,
        atlas_dir: str | Path | None = None,
        subsample_stride: int = DEFAULT_SUBSAMPLE_STRIDE,
        strict_rois: bool = True,
    ) -> None:
        """Initializes the fsaverage ROI masks.

        Args:
            roi_names: HCP-MMP parcel names to extract, without a hemisphere
                prefix (for example "V1"). Both hemispheres are merged into one
                ROI, as in the fsLR32k implementation.
            roi_groups: Optional ``{group_name: [roi names]}`` to merge after
                extraction.
            atlas_dir: Directory holding ``lh.HCP-MMP1.annot`` and
                ``rh.HCP-MMP1.annot``.
            subsample_stride: Keep every n-th vertex of each ROI. 1 keeps all.
            strict_rois: If True, a requested ROI that the atlas does not
                contain raises. If False, it is dropped with a warning (and
                removed from ``roi_names``, so downstream lookups stay valid).

        Raises:
            ValueError: If ``atlas_dir`` is missing or ``subsample_stride`` is
                not a positive integer.
        """
        super().__init__(roi_names, roi_groups)

        if atlas_dir is None:
            raise ValueError(
                "atlas_dir is required for fsaverage ROI masks; it must contain "
                "lh.HCP-MMP1.annot and rh.HCP-MMP1.annot."
            )
        self.atlas_dir = Path(atlas_dir)

        if int(subsample_stride) != subsample_stride:
            raise ValueError(
                f"subsample_stride must be an integer, got {subsample_stride}"
            )
        if subsample_stride < 1:
            raise ValueError(
                f"subsample_stride must be at least 1, got {subsample_stride}"
            )
        self.subsample_stride = int(subsample_stride)
        self.strict_rois = strict_rois

    def _read_annotations(self) -> dict[str, tuple[np.ndarray, list[str]]]:
        """Reads both hemispheres' FreeSurfer annotations.

        Returns:
            ``{hemi: (labels, names)}``. Note that with nibabel's default
            ``orig_ids=False``, ``labels`` holds *indices into the colour
            table*, not the atlas label IDs, so ROIs are matched by looking up
            a name's position in ``names``.

        Raises:
            FileNotFoundError: If either annotation file is missing.
        """
        annotations = {}
        for hemi in ("lh", "rh"):
            path = self.atlas_dir / f"{hemi}.HCP-MMP1.annot"
            if not path.exists():
                raise FileNotFoundError(
                    f"Annotation file not found: {path}. The fsaverage HCP-MMP1 "
                    "annotation can be fetched with "
                    "mne.datasets.fetch_hcp_mmp_parcellation and must be named "
                    "{lh,rh}.HCP-MMP1.annot."
                )
            labels, _, names = nib.freesurfer.io.read_annot(path)
            annotations[hemi] = (labels, [name.decode("utf-8") for name in names])
        return annotations

    def extract_rois(self) -> None:
        """Extracts per-ROI vertex indices from the fsaverage annotation.

        Raises:
            ValueError: If ``strict_rois`` and a requested ROI is not in the
                atlas, or if no requested ROI could be found at all.
        """
        annotations = self._read_annotations()
        labels_lh, names_lh = annotations["lh"]
        labels_rh, names_rh = annotations["rh"]
        n_verts_lh = len(labels_lh)

        rois: dict[str, np.ndarray] = {}
        missing: list[str] = []

        for roi_name in self.roi_names:
            hemi_indices = []
            for prefix, labels, names, offset in (
                ("L", labels_lh, names_lh, 0),
                ("R", labels_rh, names_rh, n_verts_lh),
            ):
                atlas_name = f"{prefix}_{roi_name}_ROI"
                if atlas_name not in names:
                    continue
                # labels holds colour-table indices, so match the name's position.
                label_index = names.index(atlas_name)
                indices = np.where(labels == label_index)[0]
                hemi_indices.append(indices[:: self.subsample_stride] + offset)

            if not hemi_indices:
                missing.append(roi_name)
                continue
            rois[roi_name] = np.concatenate(hemi_indices)

        if missing:
            message = (
                f"ROIs not found in the fsaverage HCP-MMP1 atlas: {missing}. "
                "Names are case-sensitive and carry no hemisphere prefix."
            )
            if self.strict_rois:
                raise ValueError(message)
            LOGGER.warning("%s Dropping them.", message)

        if not rois:
            raise ValueError(
                f"None of the requested ROIs {self.roi_names} are in the atlas."
            )

        # Drop missing ROIs from roi_names too, so that roi_names, rois,
        # voxels_per_roi and rois_to_indices stay consistent with each other.
        self.roi_names = [name for name in self.roi_names if name in rois]
        self.rois = rois
        self.voxels_per_roi = {
            roi_name: len(rois[roi_name]) for roi_name in self.roi_names
        }

        # Second pass, so the ranges are contiguous and follow roi_names order.
        # Downstream code slices concatenated per-ROI data with these.
        rois_to_indices: dict[str, tuple[int, int]] = {}
        start_idx = 0
        for roi_name in self.roi_names:
            end_idx = start_idx + self.voxels_per_roi[roi_name]
            rois_to_indices[roi_name] = (start_idx, end_idx)
            start_idx = end_idx
        self.rois_to_indices = rois_to_indices

        total_vertices = n_verts_lh + len(labels_rh)
        hcp_roi_mask = np.zeros(total_vertices, dtype=np.int_)
        for roi_indices in rois.values():
            hcp_roi_mask[roi_indices] = 1
        self.hcp_roi_mask = hcp_roi_mask

        LOGGER.info("ROIs to indices in final data array: %s", self.rois_to_indices)
        LOGGER.info("Vertex counts per ROI: %s", self.voxels_per_roi)

        if self.roi_groups is not None:
            self.merge_roi_groups(self.roi_groups)
