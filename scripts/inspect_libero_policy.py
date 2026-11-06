"""Stage 4 forward, mask, gradient, and memory checks on real samples.

Windows CMD: venv\Scripts\python.exe scripts\inspect_libero_policy.py
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from clip_features import FrozenCLIPFeatures
from libero_dataset_adapter import AlignedLiberoSequenceDataset, DEFAULT_MANIFEST
from libero_policy import LiberoActionPolicy, behavior_cloning_loss


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "stage4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.manual_seed(20261007)
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = AlignedLiberoSequenceDataset(
        DEFAULT_MANIFEST, "put_the_bowl_on_the_plate", "train"
    )
    try:
        batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)))
    finally:
        dataset.close()
    frozen = FrozenCLIPFeatures()
    policy = LiberoActionPolicy().to(frozen.device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    visual = frozen.encode_images(batch["image"])
    with torch.inference_mode():
        frozen.encode_text(batch["instruction"])  # Lightning's initial validation order.
    text, mask = frozen.encode_text(batch["instruction"])
    assert not text.is_inference()
    proprio = batch["proprio"].to(frozen.device)
    target = batch["action"].to(frozen.device)

    policy.eval()
    with torch.inference_mode():
        base = policy(visual, text, mask, proprio)
        changed_padding = text.clone()
        changed_padding[~mask.bool()] = 1000.0
        masked = policy(visual, changed_padding, mask, proprio)
        padding_effect_max_abs = float((base - masked).abs().max())
    assert padding_effect_max_abs < 1e-5, padding_effect_max_abs

    policy.train()
    prediction = policy(visual, text, mask, proprio)
    loss = behavior_cloning_loss(prediction, target)
    assert prediction.shape == target.shape == (args.batch_size, 7)
    assert bool(torch.isfinite(prediction).all() and torch.isfinite(loss))
    assert bool((prediction.abs() <= 1).all())
    loss.backward()

    groups = {
        "visual_projector": policy.visual_projector,
        "text_projector": policy.text_projector,
        "proprio_encoder": policy.proprio_encoder,
        "fusion": policy.fusion,
        "action_head": policy.action_head,
    }
    gradient_norms = {}
    for name, module in groups.items():
        gradients = [p.grad for p in module.parameters() if p.requires_grad]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        gradient_norms[name] = float(torch.linalg.vector_norm(torch.cat(
            [g.detach().flatten() for g in gradients]
        )))
        assert gradient_norms[name] > 0
    for name, parameter in (("modality", policy.modality), ("action_query", policy.action_query)):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        gradient_norms[name] = float(parameter.grad.norm())
        assert gradient_norms[name] > 0
    assert all(p.grad is None for p in frozen.model.parameters())

    report = {
        "status": "PASSED",
        "batch_size": args.batch_size,
        "visual_shape": list(visual.shape),
        "text_shape": list(text.shape),
        "mask_shape": list(mask.shape),
        "proprio_shape": list(proprio.shape),
        "output_shape": list(prediction.shape),
        "target_shape": list(target.shape),
        "loss": float(loss.detach()),
        "padding_effect_max_abs": padding_effect_max_abs,
        "trainable_parameters": sum(p.numel() for p in policy.parameters()),
        "encoder_parameter_gradients": sum(p.grad is not None for p in frozen.model.parameters()),
        "gradient_norms": gradient_norms,
        "gpu_peak_allocated_mib": (
            torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
        ),
        "gpu_peak_reserved_mib": (
            torch.cuda.max_memory_reserved() / 2**20 if torch.cuda.is_available() else None
        ),
    }
    (args.output / "inspection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
