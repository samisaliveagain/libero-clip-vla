"""Stage 3 verification of frozen CLIP features on dataset and live RGB.

Windows CMD: venv\Scripts\python.exe scripts\inspect_clip_features.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from clip_features import CHECKPOINT, FrozenCLIPFeatures, upright_live_image
from libero_dataset_adapter import AlignedLiberoSequenceDataset, DEFAULT_MANIFEST


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "stage3"
LIBERO_CHECKOUT = ROOT / "third_party" / "LIBERO"
sys.path.insert(0, str(LIBERO_CHECKOUT))
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    task = "put_the_bowl_on_the_plate"
    dataset = AlignedLiberoSequenceDataset(DEFAULT_MANIFEST, task, "train")
    try:
        sample = dataset[0]
    finally:
        dataset.close()
    image = sample["image"]
    Image.fromarray(np.rint(image.transpose(1, 2, 0) * 255).astype(np.uint8)).save(
        OUTPUT / "dataset_upright.png"
    )

    features = FrozenCLIPFeatures()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    image_batch = torch.from_numpy(image[None])
    pixels = features.preprocess_images(image_batch)
    visual = features.encode_images(image_batch)
    text, mask = features.encode_text([sample["instruction"]])
    assert visual.shape == (1, 196, 768)
    assert text.shape == (1, 77, 512)
    assert mask.shape == (1, 77)
    assert bool(torch.isfinite(visual).all() and torch.isfinite(text).all())
    assert not visual.requires_grad and not text.requires_grad
    assert all(not p.requires_grad and p.grad is None for p in features.model.parameters())

    visual_again = features.encode_images(image_batch)
    text_cached, mask_cached = features.encode_text([sample["instruction"]])
    visual_repeat_max_abs = float((visual - visual_again).abs().max())
    text_cache_max_abs = float((text - text_cached).abs().max())
    assert visual_repeat_max_abs < 1e-5
    assert text_cache_max_abs == 0 and torch.equal(mask, mask_cached)

    try:
        features.tokenize(["robot " * 100])
    except ValueError:
        overlength_rejected = True
    else:
        raise AssertionError("Overlength instruction was accepted")

    # Use a real simulator reset, with the same OpenGL-to-upright conversion
    # already established for the Stage 2 demonstration images.
    bddl = Path(get_libero_path("bddl_files")) / "libero_goal" / f"{task}.bddl"
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), robots=["Panda"], controller="OSC_POSE",
        camera_names=["agentview"], camera_heights=128,
        camera_widths=128, control_freq=20,
    )
    try:
        live = env.reset()["agentview_image"]
        live_upright = upright_live_image(live)
        Image.fromarray(live_upright).save(OUTPUT / "live_upright.png")
        live_batch = torch.from_numpy(
            live_upright.transpose(2, 0, 1).astype(np.float32)[None] / 255.0
        )
        live_visual = features.encode_images(live_batch)
        assert live_visual.shape == visual.shape
        assert bool(torch.isfinite(live_visual).all())
    finally:
        env.close()

    for _ in range(3):
        features.encode_images(image_batch)
    synchronize()
    elapsed = []
    for _ in range(10):
        start = time.perf_counter()
        features.encode_images(image_batch)
        synchronize()
        elapsed.append((time.perf_counter() - start) * 1000)

    text_elapsed = []
    for _ in range(10):
        features.text_cache.clear()  # Measure the encoder, not cache lookup.
        start = time.perf_counter()
        features.encode_text([sample["instruction"]])
        synchronize()
        text_elapsed.append((time.perf_counter() - start) * 1000)

    report = {
        "status": "PASSED",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_revision": "57c216476eefef5ab752ec549e440a49ae4ae5f3",
        "source_sample": sample["metadata"],
        "dataset_image_shape": list(image_batch.shape),
        "processed_image_shape": list(pixels.shape),
        "processed_image_finite": bool(torch.isfinite(pixels).all()),
        "vision_patch_shape": list(visual.shape),
        "text_feature_shape": list(text.shape),
        "text_mask_shape": list(mask.shape),
        "valid_text_positions": int(mask.sum()),
        "overlength_rejected": overlength_rejected,
        "encoder_trainable_parameters": sum(p.numel() for p in features.model.parameters() if p.requires_grad),
        "encoder_parameter_gradients": sum(p.grad is not None for p in features.model.parameters()),
        "vision_repeat_max_abs": visual_repeat_max_abs,
        "text_cache_max_abs": text_cache_max_abs,
        "live_vision_patch_shape": list(live_visual.shape),
        "image_latency_ms_median_batch1": statistics.median(elapsed),
        "image_latency_ms_min_batch1": min(elapsed),
        "text_latency_ms_median_batch1_uncached": statistics.median(text_elapsed),
        "gpu_peak_allocated_mib": (
            torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
        ),
        "gpu_peak_reserved_mib": (
            torch.cuda.max_memory_reserved() / 2**20 if torch.cuda.is_available() else None
        ),
    }
    (OUTPUT / "inspection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
