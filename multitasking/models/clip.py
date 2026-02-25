"""CLIP (https://github.com/openai/CLIP)."""

import clip
import einops
import PIL.Image
import torch

from .base import ImageModel


def build_model(name: str) -> ImageModel:
    """Builds a model from the CLIP codebase.

    Parameters:
        name: The name of the model to build, without the provider prefix. For example,
            "clip/RN50".
    """
    return CLIP(name)


def list_models() -> list[str]:
    """Returns all available CLIP models."""
    return clip.available_models()


class CLIP(ImageModel):
    """Official CLIP model."""

    def __init__(self, name: str):
        """Initializes the CLIP model.

        Parameters:
            name: The name of the model to build, without the provider prefix. For
                example, "RN50".
        """
        super().__init__(f"clip/{name}")

        model, preprocess = clip.load(name, device="cpu")
        self.image_encoder = model.visual
        self._preprocess_pil = preprocess

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Applies the model to a batch of images.

        Args:
            images: Images as tensor of shape (B, C, H, W), with values in the range
                [0, 255] for uint8 images and [0, 1] for float32 images.

        Returns:
            The output of the model.
        """
        if images.dtype == torch.float32:
            images = (images * 255.0).to(torch.uint8)
        elif images.dtype != torch.uint8:
            raise ValueError(f"Invalid image dtype: {images.dtype}.")

        images = torch.stack([self._preprocess(image) for image in images], dim=0)
        images = images.to(self.device)

        return self.image_encoder(images)

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        image_numpy = image.movedim(0, -1).cpu().numpy()
        image_pil = PIL.Image.fromarray(image_numpy)
        return self._preprocess_pil(image_pil)

    def postprocess(
        self,
        layer: str,
        video_shape: torch.Size,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """Postprocesses the features extracted from a layer.

        Parameters:
            layer: Name of the layer to postprocess.
            video_shape: Shape of the input video as tuple (B, C_in, T, H_in, W_in).
            feature: Features extracted from the layer as tensor of arbitrary shape.
        """
        B, _, T, _, _ = video_shape

        if layer == "image_encoder":
            # As a last step, the image encoder pools the features over all spatial
            # dimensions.
            return einops.rearrange(feature, "(B T) C -> B C T 1 1", B=B, T=T)

        if layer.startswith("image_encoder.transformer"):
            # CLIP swaps the batch and token dimensions in the vision transformer.
            return einops.rearrange(feature, "N (B T) C -> B C T N 1", B=B, T=T)

        return super().postprocess(layer, video_shape, feature)
