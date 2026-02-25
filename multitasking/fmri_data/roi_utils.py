import logging

import hcp_utils as hcp
import numpy as np

LOGGER = logging.getLogger(__name__)


class RoiMasks:
    """ROI Masks.

    Extracts brain ROI masks and allows mapping between the HCP-MMP
    (Human Connectome Project - Multimodal Parcellation) brain atlas
    and data.
    """

    def __init__(
        self,
        roi_names: list[str],
        roi_groups: dict[str, list[str]] | None = None,
    ) -> None:
        """Initialize the RoiMasks class.

        Parameters
        ----------
        roi_names : list[str]
            List of ROI names to extract from the HCP-MMP atlas.
        roi_groups: dict[str, list[str]]
            Dictionary of ROI groups to merge : {new_roi_name: [list of roi names]}, or
            None.
        """
        self.roi_names = roi_names
        self.roi_groups = roi_groups

        if roi_groups is not None:
            # Must ensure that the sub-rois are all adjacent
            #  in roi_names so that we can merge them later.
            self.roi_names = self._ensure_roi_groups_adjacent(roi_names, roi_groups)

    def _ensure_roi_groups_adjacent(
        self, roi_names: list[str], roi_groups: dict[str, list[str]]
    ):
        for group_name, gp_roi_names in roi_groups.items():
            if any(roi_name not in roi_names for roi_name in roi_names):
                raise ValueError(
                    f"A ROI out of {gp_roi_names} of group {group_name} "
                    f"not found in roi_names ({roi_names})"
                )
            # remove from list, append in the end
            roi_names = [r for r in roi_names if r not in gp_roi_names]
            roi_names.extend(gp_roi_names)

        return roi_names

    def extract_rois(self) -> None:
        hcp_roi_mask = np.zeros_like(hcp.mmp.map_all).astype(np.int_)
        roi_to_atlas_ids = {
            v: [k, k + 180]
            for k, v in hcp.mmp.labels.items()
            if ("L_" in v or "left" in v or v == "brainstem")
        }  # create roi -> atlas ID mapping from atlas ID -> roi mapping
        voxels_per_roi = dict.fromkeys(self.roi_names)
        rois_to_indices = dict.fromkeys(self.roi_names)
        rois = dict.fromkeys(self.roi_names)
        start_idx = 0
        mask_true: list[int] = []
        for roi_name in self.roi_names:
            roi_name_atlas = f"L_{roi_name}"  # L_ for left hemisphere
            roi_atlas_ids = roi_to_atlas_ids.get(
                roi_name_atlas
            )  # get the atlas IDs (one for left and right hemisphere) for the roi_name
            assert roi_atlas_ids is not None, f"no entry for {roi_name}"
            rois[roi_name] = np.where(
                (hcp.mmp.map_all == roi_atlas_ids[0])
                | (hcp.mmp.map_all == roi_atlas_ids[1])
            )[0]  # left + 180 = right
            # count the number of voxels in the roi
            voxels_per_roi[roi_name] = len(rois[roi_name])  # type: ignore
            mask_true.extend(roi_atlas_ids)
            # below assumes that ROIs are extracted from the full dataarray
            # in the same order as listed in roi_names
            rois_to_indices[roi_name] = (
                start_idx,
                start_idx + voxels_per_roi[roi_name],  # type: ignore
            )
            start_idx += voxels_per_roi[roi_name]  # type: ignore
        hcp_roi_mask[np.isin(hcp.mmp.map_all, mask_true)] = hcp.mmp.map_all[
            np.isin(hcp.mmp.map_all, mask_true)
        ]
        # LOGGER.info(f"ROIs to indices in final data array: {rois_to_indices}")
        # LOGGER.info(f"Voxel counts per ROI: {voxels_per_roi}")
        self.voxels_per_roi = voxels_per_roi
        self.hcp_roi_mask = hcp_roi_mask
        self.rois = rois
        self.roi_to_atlas_ids = {
            roi_name.split("_")[-1]: ids
            for roi_name, ids in roi_to_atlas_ids.items()
            if roi_name.split("_")[-1] in self.roi_names
        }
        self.rois_to_indices = rois_to_indices

        if self.roi_groups is not None:
            self.merge_roi_groups(self.roi_groups)

    def map_rois_to_voxels(
        self,
        response_property: np.ndarray,
    ) -> np.ndarray:
        results_array = np.zeros_like(self.hcp_roi_mask, dtype=np.float64)
        for roi_name, indices in self.rois.items():
            results_array[indices] = response_property[
                slice(*self.rois_to_indices[roi_name])  # type: ignore
            ]
        return results_array

    def merge_roi_groups(self, roi_groups: dict[str, list[str]]):
        """Merge the list of rois into a single roi with name group_name.

        Takes a dictionary {group_name: [list of roi names]} and merges the list of rois
        into a single roi with name group_name.

        We need to adjust all of:

         self.roi_names: list of names (strings)
         self.voxels_per_roi: dict of roi names -> number of voxels
         self.rois: dict of roi names -> voxel indices for that roi in the original
             full data array
         self.rois_to_indices: roi_names -> (start_idx, end_idx)

         I will not update:
         - self.roi_to_atlas_ids (dict: atlas roi name -> [left, right] atlas IDs)
         - self.hcp_roi_mask: numpy array of shape (n_voxels,)
                               since the same voxels stay active/in-use
        """
        for group_name, gp_roi_names in roi_groups.items():
            self.rois[group_name] = np.concatenate(
                [self.rois[roi] for roi in gp_roi_names]
            )
            # sanity check:
            assert len(self.rois[group_name]) == sum(  # type: ignore
                self.voxels_per_roi[roi] for roi in gp_roi_names  # type: ignore
            )
            self.voxels_per_roi[group_name] = len(self.rois[group_name])  # type: ignore
            self.rois_to_indices[group_name] = (
                self.rois_to_indices[gp_roi_names[0]][0],  # type: ignore
                self.rois_to_indices[gp_roi_names[-1]][1],  # type: ignore
            )

            # delete keys from dicts:
            for roi_name in gp_roi_names:
                del self.rois_to_indices[roi_name]
                del self.voxels_per_roi[roi_name]
                del self.rois[roi_name]

        self.roi_names = list(self.rois.keys())


def main():
    roi_names = [
        "V1",
        "V2",
        "V3",
        "V4",  # primary and early visual
        "FFC",
        "PIT",
        "V8",
        "VMV1",
        "VMV2",
        "VMV3",
        "VVC",  # ventral stream
        "IPS1",
        "V3A",
        "V3B",
        "V6",
        "V6A",
        "V7",  # dorsal stream
        "FST",
        "LO1",
        "LO2",
        "LO3",
        "MST",
        "MT",
        "PH",
        "V3CD",
        "V4t",  # MT+ complex
    ]
    # boolean_roi_mask, hcp_roi_mask, rois, rois_to_indices = extract_rois(roi_names)
    roi_masks = RoiMasks(roi_names)
    roi_masks.extract_rois()


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(fmt="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
