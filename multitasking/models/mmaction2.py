"""MMAction2 (https://github.com/open-mmlab/mmaction2)."""

import os
import tempfile
from functools import partial
from math import ceil
from pathlib import Path
from typing import Sequence
from urllib.request import urlretrieve

import einops
import imageio.v3 as iio
import torch
import torch.nn.functional as F
import torchextractor as tx
from mmaction.apis import init_recognizer
from mmaction.datasets.transforms.formatting import FormatShape, PackActionInputs
from mmaction.datasets.transforms.loading import (
    DecordDecode,
    DecordInit,
    DenseSampleFrames,
    SampleFrames,
)
from mmaction.datasets.transforms.processing import CenterCrop, Resize, ThreeCrop
from mmengine.dataset.base_dataset import Compose as MMECompose
from mmengine.dataset.utils import pseudo_collate
from net2brain.architectures.netsetbase import NetSetBase

from .base import Model

MMACTION2_PATH = "/opt/mmaction2"


_MODEL_URLS = {
    "mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb": {
        "config": f"{MMACTION2_PATH}/configs/recognition/mvit/mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb.py",  # noqa: E501
        "checkpoint": "https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/converted/mvit-small-p244_16x4x1_kinetics400-rgb_20221021-9ebaaeed.pth",
    },
    "swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb": {
        "config": f"{MMACTION2_PATH}/configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py",  # noqa: E501
        "checkpoint": "https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-241016b2.pth",
    },
    "timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb": {
        "config": f"{MMACTION2_PATH}/configs/recognition/timesformer/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb.py",  # noqa: E501
        "checkpoint": "https://download.openmmlab.com/mmaction/v1.0/recognition/timesformer/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb_20220815-a4d0d01f.pth",
    },
    "timesformer_jointST_8xb8-8x32x1-15e_kinetics400-rgb": {
        "config": f"{MMACTION2_PATH}/configs/recognition/timesformer/timesformer_jointST_8xb8-8x32x1-15e_kinetics400-rgb.py",  # noqa: E501
        "checkpoint": "https://download.openmmlab.com/mmaction/v1.0/recognition/timesformer/timesformer_jointST_8xb8-8x32x1-15e_kinetics400-rgb/timesformer_jointST_8xb8-8x32x1-15e_kinetics400-rgb_20220815-8022d1c0.pth",
    },
}

MODEL_PARAMS = {
    "mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb": {
        "preprocessor": {
            "clip_len": 16,
            "frame_interval": 4,
            "resize_size": 224,
            "crop_type": "center_crop",
            "crop_size": 224,
        },
        "extractor": {"stage": "head"},
    },
    "swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb": {
        "preprocessor": {
            "clip_len": 32,
            "frame_interval": 2,
            "resize_size": 224,
            "crop_type": "center_crop",
            "crop_size": 224,
        },
        "extractor": {"stage": "head"},
    },
    "timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb": {
        "preprocessor": {
            "clip_len": 8,
            "frame_interval": 32,
            "resize_size": 224,
            "crop_type": "center_crop",
            "crop_size": 224,
        },
        "extractor": {"stage": "head"},
    },
    "timesformer_jointST_8xb8-8x32x1-15e_kinetics400-rgb": {
        "preprocessor": {
            "clip_len": 8,
            "frame_interval": 32,
            "resize_size": 224,
            "crop_type": "center_crop",
            "crop_size": 224,
        },
        "extractor": {"stage": "head"},
    },
}


def build_model(name: str) -> Model:
    """Builds a model from the MMAction2 library.

    Parameters:
        name: The name of the model to build, without the provider prefix. For example,
            "mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb".

    Returns:
        The model instance.
    """
    return MMAction2Model(name)


def list_models() -> list[str]:
    """Returns a list of all available models."""
    return list(_MODEL_URLS.keys())


class MMAction2Model(Model):
    """A model from the MMAction2 library."""

    def __init__(self, name: str):
        """Initializes the model.

        Parameters:
            name: The name of the model to build, without the provider prefix. For
                example, "mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb".
        """
        super().__init__(f"mmaction2/{name}")

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = _MODEL_URLS[name]["config"]

            checkpoint_url = _MODEL_URLS[name]["checkpoint"]
            checkpoint_path = os.path.join(temp_dir, "checkpoint.pth")
            urlretrieve(checkpoint_url, checkpoint_path)

            self.base_model = init_recognizer(config_path, checkpoint_path)

        # This follows the config for the MViT-S model in
        # https://github.com/SergeantChris/hundred_models_brains/blob/5e802ae5e368a1a30940e6b175e47704a3061419/configs/models/mmaction/transformer.yaml#L1
        # Other models might need different parameters.
        self.preprocessor = partial(
            preprocess_mmaction, **MODEL_PARAMS[name]["preprocessor"]  # type: ignore
        )

        self.extractor = partial(
            extract_mmaction,
            model=self.base_model,
            **MODEL_PARAMS[name]["extractor"],  # type: ignore
        )

    def extract_video_features(
        self,
        video: torch.Tensor,
        layers: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        """Extracts features from a video.

        Parameters:
            video: Video as tensor of shape (B, C, T, H, W), with values in the range
                [0, 1] for float32 images and [0, 255] for uint8 images.
            layers: Layers for which to extract features.
        """
        if video.dtype == torch.float32:
            video = (video * 255.0).to(torch.uint8)

        if video.shape[0] > 1:
            raise NotImplementedError("Only a batch size of 1 is supported.")

        video = video.squeeze(0).movedim(0, -1)

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.mp4"
            iio.imwrite(video_path, video.numpy())  # type: ignore
            preprocessed_data = self.preprocessor(video_path, self.name, self.device)  # type: ignore  # noqa: E501

        features = self.extractor(preprocessed_data, layers)  # type: ignore
        features = fixed_cleaner(features)

        for key, feature in features.items():
            if feature.ndim == 3:  # B N C
                features[key] = einops.rearrange(feature, "B N C -> B C 1 N 1")
            elif feature.ndim == 5 and "swin" in self.name:
                # The SwinTransformer3D has the patch dimension last
                # https://github.com/open-mmlab/mmaction2/blob/4d6c93474730cad2f25e51109adcf96824efc7a3/mmaction/models/backbones/swin.py#L1005
                # However, the output of the extracter rather seems to return the
                # feature as (B, C, T, H, W). So something is wrong here.
                features[key] = einops.rearrange(feature, "B T H W C -> B C T H W")

        return features

    @staticmethod
    def _get_model_family(model_name: str) -> str:
        if model_name.startswith("mvit-"):
            return "mvit"
        raise ValueError(f"Unknown model family: {model_name}")


# Adapted from:
# https://github.com/SergeantChris/hundred_models_brains/blob/5e802ae5e368a1a30940e6b175e47704a3061419/repralign/models/custom_extraction_functions.py#L20-L82
def preprocess_mmaction(
    video_path,
    model_name,
    device,
    clip_len,
    frame_interval,
    resize_size,
    crop_type,
    crop_size,
    dense_sampling=False,
    format_shape="NCTHW",
):
    video = {"filename": video_path, "start_index": 0, "modality": "RGB"}
    video = DecordInit()(video)
    num_clips = ceil(video["total_frames"] / (clip_len * frame_interval))
    transform = MMECompose(
        [
            (
                SampleFrames(
                    clip_len=clip_len,
                    frame_interval=frame_interval,
                    num_clips=num_clips,
                    out_of_bound_opt="repeat_last",
                    test_mode=True,
                )
                if not dense_sampling
                else DenseSampleFrames(
                    clip_len=clip_len,
                    frame_interval=frame_interval,
                    num_clips=num_clips,
                    test_mode=True,
                )
            ),
            DecordDecode(),
            Resize(scale=(-1, resize_size)),
            (
                ThreeCrop(crop_size=crop_size)
                if crop_type == "three_crop"
                else CenterCrop(crop_size=crop_size)
            ),
            FormatShape(input_format=format_shape),
            PackActionInputs(),
        ]
    )
    video = transform(video)
    if format_shape == "NCTHW":
        # separate the clips in order to loop them in the extraction function
        video["inputs"] = (
            video["inputs"]
            .reshape(
                (-1, num_clips) + video["inputs"].shape[1:],
            )
            .transpose(0, 1)
            .contiguous()
            .float()
        )
    else:
        if clip_len == 1 and frame_interval == 1:
            num_clips = 10 if dense_sampling else 3
            if video["inputs"].size(0) % num_clips != 0:
                pad_size = (num_clips - video["inputs"].size(0) % num_clips) % num_clips
                if pad_size > num_clips / 2:
                    padded_data = video["inputs"][
                        : -(video["inputs"].size(0) % num_clips)
                    ]
                else:
                    padded_data = F.pad(
                        video["inputs"],
                        (0, 0, 0, 0, 0, 0, 0, pad_size),
                    )
                video["inputs"] = padded_data
            video["inputs"] = (
                video["inputs"]
                .reshape((-1, num_clips) + video["inputs"].shape[1:])
                .transpose(0, 1)
                .contiguous()
                .float()
            )
        else:
            video["inputs"] = video["inputs"].unsqueeze(0).float()
    return video


# Adapted from:
# https://github.com/SergeantChris/hundred_models_brains/blob/5e802ae5e368a1a30940e6b175e47704a3061419/repralign/models/custom_extraction_functions.py#L85-L118
def extract_mmaction(preprocessed_data, layers_to_extract, model, stage):
    layers = NetSetBase.select_model_layers(None, layers_to_extract, None, model)
    normalizer = model.data_preprocessor
    preprocessed_data = pseudo_collate([preprocessed_data])

    # squeeze out the fake batch
    preprocessed_data = normalizer(preprocessed_data)["inputs"].squeeze(0)

    device = preprocessed_data.device
    preprocessed_data = preprocessed_data.cpu()
    n_clips = preprocessed_data.shape[0]
    features_all_clips = {}
    for i in range(n_clips):  # sacrifice speed to avoid increasing batch size
        extractor_model = tx.Extractor(model, layers)
        try:
            out, features = extractor_model(
                preprocessed_data[i].unsqueeze(0).to(device),
                stage=stage,
            )
        except RuntimeError:
            # pad the input such as that preprocessed_data[i].shape[0] is divisible by 8
            pad_size = (8 - preprocessed_data[i].size(0) % 8) % 8
            if pad_size > 4:
                padded_data = preprocessed_data[i][
                    : -(preprocessed_data[i].size(0) % 8)
                ]
            else:
                padded_data = F.pad(
                    preprocessed_data[i],
                    (0, 0, 0, 0, 0, 0, 0, pad_size),
                )
            out, features = extractor_model(
                padded_data.unsqueeze(0).to(device),
                stage=stage,
            )
        del out
        # in mma slowfast has separate keys for slow and fast, so it doesn't need
        # special handling
        features = generic_cleaner_tuples(features)
        for key in features:
            features[key] = features[key].detach().cpu().mean(0)  # average over n_crops

            # add batch dimension (needed in next steps)
            value = features[key].unsqueeze(0)

            if key not in features_all_clips:
                features_all_clips[key] = value
            else:
                features_all_clips[key] = torch.stack(
                    [features_all_clips[key], value],
                    dim=1,
                ).mean(1)
                # average over n_clips
    return features_all_clips


# Adapted from:
# https://github.com/SergeantChris/hundred_models_brains/blob/5e802ae5e368a1a30940e6b175e47704a3061419/repralign/models/custom_extraction_functions.py#L121-L138
def fixed_cleaner(features):
    clean_dict = {}
    for A_key, subtuple in features.items():
        if isinstance(subtuple, (list, tuple)):
            if len(subtuple) >= 2:  # if subdict is a list of two values
                keys = [A_key + "_slow", A_key + "_fast"]
                for counter, key in enumerate(keys):
                    clean_dict.update({key: subtuple[counter].cpu()})
            else:
                [value] = subtuple
                clean_dict.update({A_key: value.cpu()})
        elif subtuple.shape[0] != 1:
            # this I added to cover the edge-case of giving a model a batch,
            # specifically the 2dRN model
            # it is a hack and has no place in the general pipeline
            clean_dict.update({A_key: subtuple.mean(0, keepdim=True).cpu()})
        else:
            clean_dict.update({A_key: subtuple.cpu()})
    return clean_dict


# Adapted from:
# https://github.com/SergeantChris/hundred_models_brains/blob/5e802ae5e368a1a30940e6b175e47704a3061419/repralign/models/custom_extraction_functions.py#L141-L154
def generic_cleaner_tuples(features):
    clean_dict = {}
    for A_key, subtuple in features.items():
        if isinstance(subtuple, (list, tuple)):
            tensor_elements = [elem for elem in subtuple if torch.is_tensor(elem)]
            if len(tensor_elements) == 1:
                clean_dict[A_key] = tensor_elements[0].cpu()
            else:
                new_names = [
                    A_key + f"_{counter}" for counter in range(len(tensor_elements))
                ]
                for counter, key in enumerate(new_names):
                    clean_dict[key] = tensor_elements[counter].cpu()
        else:
            clean_dict[A_key] = subtuple.cpu()
    return clean_dict
