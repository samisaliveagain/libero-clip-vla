"""Stage 3: PRETRAINED + FROZEN image/text encoders, with no robot policy.

The processor, tokenizer, token selection, and cache are NON-LEARNED
INFRASTRUCTURE. Inputs use the Stage 2 upright float32 RGB contract.
See STAGE3_AUDIT.md for the learning walkthrough and verification evidence.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = "openai/clip-vit-base-patch16"
REVISION = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
# Share the pinned download with the training/rollout encoder in clip_features.py.
MODEL_DIR = ROOT / "data" / "models" / "clip-vit-base-patch16"
PREPROCESSING_VERSION = "upright-rgb01-full224-bicubic-clip-v1"


def canonical_live_rgb(image: np.ndarray) -> torch.Tensor:
    """Raw LIBERO OpenGL uint8 HWC -> upright float32 CHW, exactly once.

    Dataset samples ALREADY have this convention; do not call this on them.
    The live renderer must use robosuite's verified 'opengl' convention.
    """
    if image.dtype != np.uint8 or image.shape != (128, 128, 3):
        raise ValueError("Expected raw LIBERO uint8 RGB [128,128,3].")
    upright = np.flip(image, axis=0).transpose(2, 0, 1).copy()
    return torch.from_numpy(upright).float().div(255.0)


class FrozenCLIPEncoder(nn.Module):
    """Keep contextual patches/text; omit CLIP's pooled similarity heads.

    Both encoders stay in eval mode even when a parent calls train(). No
    project gradients enter them. no_grad (rather than inference_mode)
    returns ordinary detached tensors usable by later trainable components.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        if not (MODEL_DIR / "pytorch_model.bin").is_file():
            raise FileNotFoundError("Run scripts/download_clip_stage3.py first.")
        source = dict(local_files_only=True)
        self.processor = CLIPImageProcessor.from_pretrained(MODEL_DIR, **source)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(MODEL_DIR, **source)
        clip = CLIPModel.from_pretrained(
            MODEL_DIR, **source, use_safetensors=False,
            torch_dtype=torch.float32, attn_implementation="sdpa",
        )
        self.vision = clip.vision_model
        self.text = clip.text_model
        self.requires_grad_(False)
        self.train(False)
        self.to(device)
        # CPU FP32 cache is instance-local: checkpoint/tokenizer are fixed.
        # Reuse is independent of the GPU device and caller's autocast mode.
        self._text_cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self.cache_capacity = 128

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def preprocess_images(self, images: torch.Tensor | np.ndarray) -> torch.Tensor:
        """[B,3,128,128] upright RGB [0,1] -> CPU [B,3,224,224]."""
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()
        images = np.asarray(images)
        if images.ndim != 4 or images.shape[1:] != (3, 128, 128) or len(images) == 0:
            raise ValueError("Expected nonempty upright RGB [B,3,128,128].")
        if not np.issubdtype(images.dtype, np.floating):
            raise ValueError("Expected float RGB already scaled to [0,1].")
        if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("RGB must be finite and in [0,1].")
        return self.processor(
            images=list(images.astype(np.float32, copy=False)),
            input_data_format="channels_first", data_format="channels_first",
            do_rescale=False, do_center_crop=False,
            size={"shortest_edge": 224}, return_tensors="pt",
        )["pixel_values"]

    def tokenize(self, instructions: list[str]) -> dict[str, torch.Tensor]:
        """Return IDs and tokenizer-derived validity; reject silent truncation."""
        if isinstance(instructions, str) or not instructions or not all(
            isinstance(text, str) for text in instructions
        ):
            raise ValueError("Expected a nonempty list of instruction strings.")
        # Includes BOS/EOS. An empty instruction is valid for later ablations.
        unpadded = self.tokenizer(instructions, padding=False, truncation=False)
        lengths = [len(ids) for ids in unpadded["input_ids"]]
        if max(lengths) > 77:
            raise ValueError(f"Instruction exceeds 77 tokens including BOS/EOS: {lengths}")
        return dict(self.tokenizer(
            instructions, padding="max_length", max_length=77,
            truncation=False, return_attention_mask=True, return_tensors="pt",
        ))

    @torch.no_grad()
    def encode_pixels(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Preprocessed image -> 196 contextual patch tokens, excluding CLS."""
        # Caller may use autocast for vision. We keep FP32 weights by default.
        hidden = self.vision(pixel_values=pixel_values.to(self.device)).last_hidden_state
        return hidden[:, 1:, :].contiguous()

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor | np.ndarray) -> torch.Tensor:
        return self.encode_pixels(self.preprocess_images(images))

    @torch.no_grad()
    def encode_text(self, instructions: list[str], *, use_cache: bool = True
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Validate even on cache hits. This also rejects a string used as a list.
        tokens = self.tokenize(instructions)
        missing = list(dict.fromkeys(
            text for text in instructions if not use_cache or text not in self._text_cache
        ))
        fresh = {}
        if missing:
            encoded = self.tokenize(missing)
            inputs = {key: value.to(self.device) for key, value in encoded.items()}
            # Stable cache precision, even inside a future policy's autocast.
            with torch.autocast(device_type=self.device.type, enabled=False):
                features = self.text(**inputs).last_hidden_state
            for i, text in enumerate(missing):
                fresh[text] = (features[i].float().cpu(), encoded["attention_mask"][i].bool())
        # Assemble before eviction so a large request can exceed cache capacity.
        entries = [fresh[text] if text in fresh else self._text_cache[text]
                   for text in instructions]
        if use_cache:
            for text, entry in zip(instructions, entries):
                self._text_cache[text] = entry
                self._text_cache.move_to_end(text)
            while len(self._text_cache) > self.cache_capacity:
                self._text_cache.popitem(last=False)
        features = torch.stack([entry[0] for entry in entries]).to(self.device)
        mask = torch.stack([entry[1] for entry in entries]).to(self.device)
        if not torch.equal(mask.cpu(), tokens["attention_mask"].bool()):
            raise RuntimeError("Cached text validity does not match the tokenizer.")
        return features, mask

    def clear_text_cache(self) -> None:
        self._text_cache.clear()

    def forward(self, images: torch.Tensor | np.ndarray, instructions: list[str]) -> dict:
        if len(images) != len(instructions):
            raise ValueError("Image and instruction batch sizes differ.")
        visual = self.encode_images(images)
        text, mask = self.encode_text(instructions)
        return {"vision_tokens": visual, "text_tokens": text, "text_mask": mask}
