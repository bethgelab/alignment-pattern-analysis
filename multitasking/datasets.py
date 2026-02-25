import logging
import os
import pickle
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal, Sequence

import imageio.v3 as iio
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from multitasking.fmri_data.roi_utils import RoiMasks

LOGGER = logging.getLogger(__name__)

SCRATCH = Path(os.environ.get("SCRATCH", "/scratch"))


class FMRIObjaverse(Dataset):
    """Loads videos from the fMRI-Objaverse dataset."""

    def __init__(
        self,
        path: str | os.PathLike,
        sample_ids: Sequence[str],
        resolution: int | tuple[int, int] | None = None,
        frame_rate: float | None = None,
    ):
        """Initializes the dataset.

        Parameters:
            path: Path to the dataset archive (zip).
            resolution: If provided, the videos will be resized to the given
                resolution.
            frame_rate: If provided, the videos will be resampled to the given
                frame rate.
        """
        self.path = Path(path).resolve()

        if self.path.name.endswith(".zip"):
            target_path = SCRATCH / "fmri_objaverse"
            if not target_path.exists():
                LOGGER.info(f"Unzipping dataset archive to {target_path}")
                with zipfile.ZipFile(self.path, "r") as archive:
                    archive.extractall(target_path)
            else:
                LOGGER.info(f"Dataset archive already unzipped to {target_path}")
        else:
            target_path = self.path

        self.videos = [
            target_path / "stimuli" / f"{sample_id}.mp4" for sample_id in sample_ids
        ]

        if isinstance(resolution, int):
            self.resolution = (resolution, resolution)
        elif isinstance(resolution, tuple):
            if len(resolution) != 2 or resolution[0] != resolution[1]:
                raise ValueError(f"Invalid resolution: {resolution}")
            self.resolution = resolution
        else:
            self.resolution = None  # type: ignore

        self.frame_rate = frame_rate

    def __len__(self):
        """Returns the number of sample in the dataset."""
        return len(self.videos)

    def __getitem__(self, index: int) -> dict[str, str | torch.Tensor]:
        """Returns a sample from the dataset.

        Returns:
            Dictionary with the following keys:
                - "id": The ID of the video.
                - "video": The video as a uint8 tensor of shape (C, T, H, W).
        """
        video_path = self.videos[index]
        return {
            "id": video_path.stem,
            "video": self._load_video(video_path),
        }

    def _load_video(self, video_path: Path) -> torch.Tensor:
        filter_sequence = []

        if self.frame_rate is not None:
            filter_sequence.append(("framerate", f"fps={self.frame_rate}"))

        if self.resolution is not None:
            filter_sequence.append(
                ("scale", f"w={self.resolution[0]}:h={self.resolution[1]}")
            )

        video = iio.imread(video_path, plugin="pyav", filter_sequence=filter_sequence)
        return torch.from_numpy(video).movedim(-1, 0)


class BoldMoments:
    """Loads videos from the fMRI-Objaverse dataset."""

    def __init__(
        self,
        video_path: str | os.PathLike,
        fmri_path: str | os.PathLike,
        rois: list[str],
        split: Literal["train", "test"],
        subject_ids: str | list[str] = "all",
        aggregate_fn: str = "avg",
        resolution: int | tuple[int, int] | None = None,
        frame_rate: float | None = None
    ):
        """Initializes the BoldMoments dataset.

        Args:
            video_path: Path to the directory containing the video files.
            fmri_path: Path to the directory containing the fMRI data.
            subject_ids: Subject ID or list of subject IDs to load.
            rois: List of region-of-interest (ROI) names.
            split: Which split to use, either "train" or "test".
            aggregate_fn: Aggregation function to use for fMRI data
                (default: "avg"; if set to None, keep the second dimension
                (dim 1) as trial dimension).
            resolution: If provided, the videos will be resized to the given
                resolution (int or (width, height) tuple).
            frame_rate: If provided, the videos will be resampled to the given
                frame rate.
        """
        self.video_path = Path(video_path).resolve()
        self.fmri_path = Path(fmri_path).resolve()
        video_target_path = self.video_path / Path("mp4_h264")
        self.all_sample_ids = [
            f.name.split(".")[0] for f in video_target_path.glob("*.mp4")
        ]
        self.all_sample_ids.sort()
        self.all_sample_ids_by_split = {
            "train": self.all_sample_ids[:1000],
            "test": self.all_sample_ids[1000:],
        }
        self.invalid_sample_ids = {
            "train": ["0258", "0831"],
            "test": ["1001", "1097", "1090", "1085", "1094", "1093"],
        }

        self.valid_sample_ids = {
            split: [
                sample_id
                for sample_id in self.all_sample_ids_by_split[split]
                if sample_id not in self.invalid_sample_ids[split]
            ]
            for split in ["train", "test"]
        }
        self.videos = {
            split: [
                video_target_path / f"{sample_id}.mp4"
                for sample_id in self.valid_sample_ids[split]
            ]
            for split in ["train", "test"]
        }
        if subject_ids == "all":
            self.subject_ids = [
                p.stem
                for p in self.fmri_path.iterdir()
                if p.is_dir() and "sub" in p.name
            ]
            self.subject_ids.sort()  # to avoid confusion
        else:
            self.subject_ids = list(subject_ids)
        self.rois = rois
        self.roi_masks = RoiMasks(rois)
        self.roi_masks.extract_rois()
        if aggregate_fn == "avg":
            self.aggregate_fn = np.mean
        else:
            raise NotImplementedError(
                f"aggregate_fn {aggregate_fn} is not implemented."
            )
        self.split = split

        LOGGER.info(
            f"Loading {len(self.videos['train'])} train videos and "
            f"{len(self.videos['test'])} test videos from {video_target_path}"
        )

        if isinstance(resolution, int):
            self.resolution = (resolution, resolution)
        elif isinstance(resolution, tuple):
            if len(resolution) != 2 or resolution[0] != resolution[1]:
                raise ValueError(f"Invalid resolution: {resolution}")
            self.resolution = resolution
        else:
            self.resolution = None  # type: ignore

        self.frame_rate = frame_rate

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.videos[self.split])

    def __getitem__(self, index: int) -> dict[str, str | torch.Tensor]:
        """This function is ambiguous, hence not used.

        Legacy:
        Returns a sample from the dataset.

        Returns:
            Dictionary with the following keys:
                - "id": The ID of the video.
                - "video": The video as a uint8 tensor of shape (C, T, H, W).
        """
        raise NotImplementedError
        # video_path = self.videos[index]

        # return {
        #     "id": video_path.stem,
        #     "video": self._load_video(video_path),
        # }

    def load_fmri_data(self):
        """Loads fMRI data for all subjects, current split, and ROIs from file.

        This method iterates over all subject IDs specified in
        `self.subject_ids` and loads the corresponding fMRI data for each
        subject for the current split (e.g., "train" or "test").
        The data is loaded from preprocessed files on disk, reformatted
        ROI-wise, and returned as a 2-tier dictionary mapping subject IDs to
        their ROI-wise fMRI data.

        Returns:
            dict[str, dict[str, np.ndarray]]:
                A dictionary where each key is a subject ID and each value is
                another dictionary mapping ROI names to their corresponding
                fMRI data arrays for that subject.
        """
        fmri_data = {sub: None for sub in self.subject_ids}

        for sub in self.subject_ids:
            fmri_data[sub] = self._reformat_data_roi_wise(
                self._load_fmri_per_subject(sub, return_sample_ids=False)
            )
        return fmri_data

    def load_videos(self, batch_size=1) ->  \
        Iterator[tuple[torch.Tensor, list[str]]]:
        """Loads all videos for the current split as an iterator of tensors.

        This method iterates over the list of video file paths corresponding to
        the current split (e.g., "train" or "test") and yields each video as a
        tensor. The videos are loaded using the internal `_load_video` method.

        Yields:
            torch.Tensor: The video tensor for each video in the current split.
        """
        batch = []
        ids = []
        for video_path in self.videos[self.split]:
            video_tensor = self._load_video(video_path)  # load one video tensor
            batch.append(video_tensor)
            ids.append(video_path.stem.split(".")[0])
            if len(batch) == batch_size:
                yield (torch.stack(batch), ids)  # stack into one batch tensor
                batch = []
                ids = []
        # yield any remaining videos if they don't fill a full batch
        if batch:
            yield (torch.stack(batch), ids)

    def _load_fmri_per_subject(self, subject_id: str, return_sample_ids: bool = False):
        """Loads and returns the valid fMRI data for a given subject.

        The call to _clean_samples ensures fMRI data points are cleaned from
        invalid sample IDs, and are returned in the order specified in
        self.valid_sample_ids.

        Args:
            subject_id (str): The ID of the subject whose fMRI data is to be
                loaded.
            return_sample_ids (bool, optional): If True, also returns the list
                of sample IDs corresponding to the fMRI data points.
                Defaults to False.

        Returns:
            NDArray[np.float64] or Tuple[NDArray[np.float64], list[str]]:
                - If return_sample_ids is False, returns the cleaned fMRI data
                    array.
                - If return_sample_ids is True, returns a tuple of the cleaned
                    fMRI data array and the list of cleaned sample IDs.
        """
        with open(
            self.fmri_path
            / f"{subject_id}/prepared_betas/{subject_id}_organized_betas_task-{
                self.split
            }_normalized.pkl",
            "rb",
        ) as f:
            fmri_data, sample_ids = pickle.load(f)

        clean_fmri_data, clean_sample_ids = self._clean_samples(fmri_data, sample_ids)
        clean_fmri_data = self.aggregate_fn(clean_fmri_data, axis=1)
        assert clean_fmri_data.ndim == 2, (
            "Expected aggregated fMRI data to be of shape 2, got {clean_fmri_data.ndim}"
        )
        if return_sample_ids:
            return clean_fmri_data, clean_sample_ids
        else:
            return clean_fmri_data

    def _clean_samples(self, fmri_data: NDArray[np.float64], sample_ids: list[str]):
        """Selects all valid fMRI data points within the current split.

        An fMRI data point is a [1, n_reps, n_voxels] entry in a Numpy Array.
        Valid data points are all data points that are not specified invalid
        in the BOLDMoment's __init__ function.

        Args:
            fmri_data (NDArray[np.float64]): The fMRI data array, where each
                entry is a data point
            sample_ids (list[str]): List of sample IDs corresponding to the fMRI
                data points

        Returns:
            Tuple[NDArray[np.float64], list[str]]:
                - The cleaned fMRI data array containing only valid samples.
                - The list of cleaned sample IDs corresponding to the valid
                    samples.

        Raises:
            AssertionError: If the cleaned sample IDs do not match the expected
                valid sample IDs for the current split.
        """
        clean_fmri_data, clean_sample_ids = map(
            list,
            zip(
                *(
                    (data_point, sid[3:])
                    for data_point, sid in zip(fmri_data, sample_ids, strict=True)
                    if sid[3:]  # IDs look like "vid0001"
                    in self.valid_sample_ids[self.split]
                ),
                strict=True,
            ),
        )
        assert self.valid_sample_ids[self.split] == clean_sample_ids, (
            "fMRI samples don't match valid_samples!"
        )
        return np.asarray(clean_fmri_data), clean_sample_ids

    def _reformat_data_roi_wise(self, fmri_data: NDArray[np.float64]):
        """Reformats the fMRI data array to be ROI-wise.

        This method takes the full fMRI data array and splits it into separate
        arrays for each region of interest (ROI), using the ROI masks. The
        output is a dictionary mapping each ROI name to its corresponding fMRI
        data subset.

        Args:
            fmri_data (NDArray[np.float64]): The fMRI data array of shape
                [n_samples, n_reps, n_voxels].

        Returns:
            dict[str, NDArray[np.float64]]: A dictionary where each key is a ROI
                name and each value is the fMRI data array for that ROI, with
                shape [n_samples, n_reps, n_voxels_in_roi].
        """
        return {
            roi: fmri_data[..., self.roi_masks.rois[roi]]
            for roi in self.roi_masks.roi_names
        }

    def _load_video(self, video_path: Path) -> torch.Tensor:
        filter_sequence = []

        max_frames = None
        if self.frame_rate is not None:
            filter_sequence.append(("framerate", f"fps={self.frame_rate}"))
            max_frames = int(np.ceil(3 * self.frame_rate))  # 3 seconds of video

        if self.resolution is not None:
            filter_sequence.append(
                ("scale", f"w={self.resolution[0]}:h={self.resolution[1]}")
            )

        try:
            video = iio.imread(
                video_path,
                plugin="pyav",
                filter_sequence=filter_sequence,
            )
        except Exception as e:
            LOGGER.info(f"Error loading video {video_path}: {e}")

            LOGGER.error(f"Error loading video {video_path}: {e}")
            LOGGER.error(
                "This might be due to a missing video file or an unsupported format."
            )
            raise

        # if dim 2 of video is larger than max_frames, delete the last frames
        if max_frames is not None and video.shape[0] > max_frames:
            video = video[:max_frames, :, :, :]
        # if there is less

        return torch.from_numpy(video).movedim(-1, 0)


def build_dataset(
    config: dict[str, Any], split: Literal["train", "test"]
) -> BoldMoments:
    """Builds a dataset from a config."""
    if config["dataset"].get("name") == "bold_moments":
        subject_ids = config["fmri"].get("sub_id", "all")
        if type(subject_ids) is str and subject_ids != "all":
            subject_ids = [subject_ids]
        return BoldMoments(
            video_path=config["dataset"]["path"],
            fmri_path=config["fmri"]["path"],
            subject_ids=subject_ids,
            rois=config["fmri"]["roi_names"],
            split=split,
            resolution=config["dataset"].get("resolution", None),
            frame_rate=config["dataset"].get("frame_rate", None),
        )

    elif config.get("name") == "fmri_objaverse":
        raise NotImplementedError("fMRI-Objaverse is no longer supported.")

    else:
        raise ValueError(f"Invalid dataset name: {config.get('name')}")
