"""Frozen CLIP token features for Stage 3.

Input images follow the Stage 2 adapter contract: upright RGB float32
[B,3,128,128] in [0,1]. Live LIBERO images must first be flipped upright.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "data" / "models" / "clip-vit-base-patch16"


def upright_live_image(image: np.ndarray) -> np.ndarray:
    """Convert LIBERO's raw OpenGL RGB image to the Stage 2 image convention."""
    if image.shape != (128, 128, 3) or image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 live RGB [128,128,3], got {image.shape} {image.dtype}")
    return np.flip(image, axis=0).copy()


class FrozenCLIPFeatures:
    """Return spatial vision and contextual text tokens, without training CLIP."""

    def __init__(self, checkpoint: Path = CHECKPOINT, device: str = "cuda"):
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        self.processor = CLIPImageProcessor.from_pretrained(self.checkpoint, local_files_only=True)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(self.checkpoint, local_files_only=True)
        self.model = CLIPModel.from_pretrained(self.checkpoint, local_files_only=True).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        if (self.model.config.vision_config.image_size,
                self.model.config.vision_config.patch_size,
                self.model.config.text_config.max_position_embeddings) != (224, 16, 77):
            raise ValueError("Checkpoint shape contract differs from the selected architecture")

    def preprocess_images(self, upright_rgb: torch.Tensor | np.ndarray) -> torch.Tensor:
        images = torch.as_tensor(upright_rgb, dtype=torch.float32)
        if images.ndim != 4 or images.shape[1:] != (3, 128, 128):
            raise ValueError(f"Expected upright RGB [B,3,128,128], got {tuple(images.shape)}")
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("Input pixels must be finite and scaled to [0,1]")
        result = self.processor(
            images=list(images),
            do_rescale=False,  # Stage 2 already divided uint8 pixels by 255.
            do_center_crop=False,  # Preserve the complete square camera view.
            return_tensors="pt",
        )["pixel_values"]
        if result.shape != (images.shape[0], 3, 224, 224):
            raise AssertionError(f"Unexpected processed image shape: {tuple(result.shape)}")
        return result.to(self.device)

    def tokenize(self, instructions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        for instruction in instructions:
            if len(self.tokenizer(instruction, add_special_tokens=True)["input_ids"]) > 77:
                raise ValueError(f"Instruction exceeds CLIP's 77-token context: {instruction!r}")
        tokenized = self.tokenizer(
            instructions, padding="max_length", max_length=77,
            truncation=False, return_tensors="pt",
        )
        return tokenized["input_ids"].to(self.device), tokenized["attention_mask"].to(self.device)

    @torch.no_grad()
    def encode_images(self, upright_rgb: torch.Tensor | np.ndarray) -> torch.Tensor:
        pixels = self.preprocess_images(upright_rgb)
        hidden = self.model.vision_model(pixel_values=pixels).last_hidden_state
        if hidden.shape[1:] != (197, 768):
            raise AssertionError(f"Unexpected vision tokens: {tuple(hidden.shape)}")
        return hidden[:, 1:, :]  # Keep 14 x 14 spatial patches; omit CLS.

    @torch.no_grad()
    def encode_text(self, instructions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        unseen = list(dict.fromkeys(text for text in instructions if text not in self.text_cache))
        if unseen:
            ids, mask = self.tokenize(unseen)
            hidden = self.model.text_model(input_ids=ids, attention_mask=mask).last_hidden_state
            if hidden.shape[1:] != (77, 512):
                raise AssertionError(f"Unexpected text tokens: {tuple(hidden.shape)}")
            for text, features, validity in zip(unseen, hidden, mask):
                # Lightning may run validation in inference_mode before the
                # first training batch. Cache ordinary detached tensors so
                # the trainable text projector can save them for backward.
                with torch.inference_mode(False):
                    self.text_cache[text] = (features.detach().clone(), validity.detach().clone())
        features = torch.stack([self.text_cache[text][0] for text in instructions])
        masks = torch.stack([self.text_cache[text][1] for text in instructions])
        return features, masks
