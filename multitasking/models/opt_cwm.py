"""Opt-CWM (https://github.com/neuroailab/Opt_CWM/)."""

from pathlib import Path
from typing import Sequence

import torch

from ._extern.opt_cwm.models import builder
from ._extern.opt_cwm.utils import constants, options, utils
from .base import LayerOutputRecorder, Model

BASE_CONFIG = Path(__file__).parent.resolve() / "_extern/opt_cwm/configs/eval_cfg.yaml"


def build_model(name: str = "opt_cwm") -> Model:
    """Builds the Opt-CWM model.

    Parameters:
        name: Name of the model to load. Currently only "opt_cwm" is supported.
    """
    return OptCWM(name)


def list_models() -> list[str]:
    """Returns a list of all available models."""
    return ["opt_cwm"]


class OptCWM(Model):
    def __init__(self, name: str = "opt_cwm"):
        """Initializes the Opt-CWM model.

        Parameters:
            name: Name of the model to load.
        """
        if name != "opt_cwm":
            raise ValueError(
                f"Invalid model name: {name}. Only 'opt_cwm' is supported."
            )

        super().__init__(f"opt_cwm/{name}")

        args = [
            f"--yaml={BASE_CONFIG}",
            "--model_args.flow_predictor.masking_iters=1",
            "--model_args.flow_predictor.zoom_iters=4",
        ]

        opt_cmd = options.parse_arguments(args)
        eval_cfg = options.set(opt_cmd=opt_cmd, verbose=False)
        model_args = eval_cfg.model_args

        self.base_model = builder.get_flow_predictor(model_args)
        self.base_model.load_pretrained(
            model_args.build.highres,
            model_args.build.force,
        ).requires_grad_(False)

    @property
    def layers(self) -> list[str]:
        """Layers of the model that can be extracted."""
        list_layers = ["cwm_model.encoder.blocks.0",
                       "cwm_model.encoder.blocks.1",
                       "cwm_model.encoder.blocks.2",
                       "cwm_model.encoder.blocks.3",
                       "cwm_model.encoder.blocks.4",
                       "cwm_model.encoder.blocks.5",
                       "cwm_model.encoder.blocks.6",
                       "cwm_model.encoder.blocks.7",
                       "cwm_model.encoder.blocks.8",
                       "cwm_model.encoder.blocks.9",
                       "cwm_model.encoder.blocks.10",
                       "cwm_model.encoder.blocks.11"]
        return list_layers

    def extract_video_features(
        self,
        video: torch.Tensor,
        layers: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        """Extracts features from a video.

        Parameters:
            video: Video to extract features from as tensor of shape (B, C, T, H, W).
                Accepts float32 in the range [0, 1] or uint8.
            layers: Layers for which to extract features. For this model, only
                "cwm_model.encoder" is supported.

        Returns:
            Dictionary with the requested layer outputs.
        """
        with LayerOutputRecorder(self.base_model, layers) as recorder:
            self.forward(video)
            outputs = recorder.outputs

        features = {}

        for layer_name, layer_outputs in outputs.items():
            # add two empty dimensions to the end
            layer_outputs = layer_outputs.unsqueeze(0).unsqueeze(-1)  # NCTHW
            # from N, T, C, H, W, to N, C, T, H, W
            layer_outputs = layer_outputs.movedim(1, 2)  # NCTHW -> NCTHW

            features[layer_name] = layer_outputs



        return features

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Opt-CWM encoder.

        Args:
            video: Video as tensor of shape (B, C, T, H, W), with values in the range
                [0, 255] for uint8 images and [0, 1] for float32 images.

        Returns:
            The output of the Opt-CWM encoder as tensor of shape (B, C', T, H', W').
        """
        video = self._preprocess(video)
        B, _, T, _, _ = video.shape

        # The Opt-CWM model can only process pairs of frames. We always pair each frame
        # with the next frame, using cyclic boundary conditions.
        next_frame = torch.cat([video[:, :, 1:], video[:, :, 0:1]], dim=2)
        frame_pairs = torch.stack([video, next_frame], dim=3)  # B, C, T, 2, H, W
        frame_pairs = frame_pairs.movedim(2, 1)  # B, T, C, 2, H, W
        frame_pairs = frame_pairs.flatten(0, 1)  # (B * T), C, 2, H, W

        # The mask gets in inverted -> 1 = masked, 0 = use input patch
        H, W = self.base_model.n_patches
        mask = torch.zeros(1, 2 * H * W, dtype=torch.bool, device=video.device)
        mask = mask.expand(B * T, -1)

        _, encoder_out = self.base_model.cwm_model(
            frame_pairs,
            mask=mask,
            get_encoder_out=True,
            res=self.base_model.cwm_model._pos_emb_scale,
            res_y=self.base_model.cwm_model._pos_emb_scale,
        )

        encoder_out = encoder_out.view(B * T, 2, H, W, -1)
        encoder_out = encoder_out[:, 0].view(B, T, H, W, -1)
        encoder_out = encoder_out.movedim(-1, 1)

        return encoder_out

    def _preprocess(self, video: torch.Tensor) -> torch.Tensor:
        """Video preprocessing.

        The preprocesing is adapted from the `self.base_model._preproc_video` method,
        but omits preprocessing the pixel locations. It additionally includes conversion
        to float32 and normalization using the statistics of the ImageNet dataset.
        """
        B, C, T, H, W = video.shape

        if video.dtype == torch.uint8:
            video = video.float() / 255.0
        elif video.dtype != torch.float32:
            raise ValueError(
                f"Input must be of type uint8 or float32, got {video.dtype}."
            )

        video = video.to(self.device)

        mean = torch.tensor(constants.IMAGENET_DEFAULT_MEAN).to(self.device)
        mean = mean[None, :, None, None, None]
        std = torch.tensor(constants.IMAGENET_DEFAULT_STD).to(self.device)
        std = std[None, :, None, None, None]
        video = (video - mean) / std

        if (H, W) == self.base_model.input_size:
            return video

        video = utils.batch_resize_video(video, self.base_model.input_size)
        utils.size_guard(video, (B, C, T, *self.base_model.input_size))
        return video

