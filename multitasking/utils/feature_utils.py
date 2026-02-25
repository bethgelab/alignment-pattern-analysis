from typing import Dict

import numpy as np
import torch
from sklearn import preprocessing


def get_splithalf_xy(
    model_features: np.ndarray, fmri_data: np.ndarray, split_indices: Dict
) -> Dict:
    """Creates 50/50 train test splits at the object level.

    Adapted from https://github.com/ColinConwell/DeepNSD/blob/fc461d021785c341fa1b0ce9a3dd794d5df02d12/source_code/pressures/main_analysis.py#L13
    """
    n_samples, n_frames, n_features = model_features.shape
    n_voxels = fmri_data.shape[-1]
    data_splits: dict[str, dict[str, np.ndarray]] = {"train": {}, "test": {}}

    scaler = preprocessing.StandardScaler()
    data_splits["train"]["X"] = scaler.fit_transform(
        model_features[split_indices["train"], ...].reshape(
            (-1, n_features),
            order="C",  # flatten objects * frames
        )
    )
    data_splits["test"]["X"] = scaler.transform(
        model_features[split_indices["test"], ...].reshape((-1, n_features), order="C")
    )

    data_splits["train"]["Y"] = fmri_data[split_indices["train"], ...].reshape(
        (-1, n_voxels), order="C"
    )
    data_splits["test"]["Y"] = fmri_data[split_indices["test"], ...].reshape(
        (-1, n_voxels), order="C"
    )

    return data_splits


def reshape_model_features(
    features: np.ndarray,
    fmri_framerate: int = 10,
) -> np.ndarray:
    """Reshapes the model features.

    Does all the slicin' and dicin' to turn model features into design matrices
    matching the number of (valid) samples in the fMRI data and the framerate
    """
    # check if features are nd array and if not, convert to numpy array
    if not isinstance(features, np.ndarray):
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        else:
            raise ValueError("features must be a numpy array or a torch tensor")

    n_samples, n_frames, height, width, n_feature_channels = (
        features.shape
    )  # n_samples, n_features, frames [5 frames/s x 6.4 s], height, width
    n_features = (
        n_feature_channels * height * width
    )  # concatenate spatial and feature channel dimensions
    features = np.moveaxis(features, -1, 2)  # move feature channels to axis 2
    # -> (n_samples, n_frames, n_feature_channels, height, width)
    features = np.reshape(
        features, (n_samples, n_frames, n_features), order="C"
    )  # Fixed line

    assert features.shape[1] == fmri_framerate, (
        "Features and fMRI framerate don't match"
    )

    return features
