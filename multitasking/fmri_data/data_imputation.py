import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


def impute_nans(fmri_data: np.ndarray, logging_roi_name, logging_split) -> np.ndarray:
    """Impute NaNs in the fmri data.

    fmri_data: (n_samples, n_trials, n_voxels) (ndarray)

    Try to fill the nan values first with trial means, then
    the remaining nans with zeros.

    The trial-mean first step is unneccessary for our data because we
    take a trial-nanmean beforehand anyways, but this is more generic
    and also works if we decide to use the trials differently.
    """
    if np.any(np.isnan(fmri_data)):
        LOGGER.warning(f"ROI {logging_roi_name} has NaNs (split: {logging_split})!")
        # import ipdb; ipdb.set_trace()
        LOGGER.warning(np.where(np.isnan(fmri_data)))
        LOGGER.warning(f"What voxels: {np.unique(np.where(np.isnan(fmri_data))[2])}")
        LOGGER.warning("I will try to impute trial means first")

        nan_ids = np.where(np.isnan(fmri_data))
        trial_mean = np.nanmean(fmri_data, axis=1)
        fmri_data[nan_ids] = trial_mean[nan_ids[0], nan_ids[2]]
        n_imputed = np.sum(~np.isnan(fmri_data[nan_ids]))
        LOGGER.warning(f"Imputed {n_imputed} voxels with trial means")

        nan_ids = np.where(np.isnan(fmri_data))
        if len(nan_ids[0]) > 0:
            # Impute zeros
            LOGGER.warning(
                f"{len(np.unique(nan_ids[0]))} of "
                f"{fmri_data.shape[0]} samples and "
                f"{len(np.unique(nan_ids[2]))} of "
                f"{fmri_data.shape[2]} voxels had nans "
                "for all trials, imputing zeros..."
            )
            LOGGER.warning(f"What voxels: {np.unique(nan_ids[2])}")
            fmri_data[nan_ids] = 0

    return fmri_data
