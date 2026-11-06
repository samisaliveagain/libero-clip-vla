"""Stage 3 verification: real RGB/text -> frozen tokens, never robot actions.

NON-LEARNED INFRASTRUCTURE. Uses training-split frames only, original HDF5
read-only, and the existing Stage 2 simulator reconstruction helpers.
Run: venv\Scripts\python.exe scripts\inspect_clip_representations.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np
from PIL import Image, ImageDraw
import torch

from frozen_clip_encoder import (
    CHECKPOINT, MODEL_DIR, PREPROCESSING_VERSION, REVISION, ROOT,
    FrozenCLIPEncoder, canonical_live_rgb,
)
from download_clip_stage3 import WEIGHTS_SHA256
from libero_dataset_adapter import AlignedLiberoSequenceDataset, DEFAULT_MANIFEST

OUTPUT = ROOT / "outputs" / "stage3"


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stats(tensor):
    x = tensor.detach().float()
    check(torch.isfinite(x).all().item(), "Non-finite feature or pixel value")
    return {"shape": list(x.shape), "dtype": str(tensor.dtype),
            "min": x.min().item(), "max": x.max().item(),
            "mean": x.mean().item(), "std": x.std().item(),
            "requires_grad": tensor.requires_grad}


def rejects(function):
    try:
        function()
    except ValueError:
        return True
    raise AssertionError("Invalid input was accepted")


def gpu_snapshot():
    try:
        value = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
             "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
        name, total, used, driver = [part.strip() for part in value.split(",")]
        return {"name": name, "total_mib": int(total), "used_mib": int(used),
                "driver": driver, "scope": "whole GPU, includes other applications"}
    except (OSError, subprocess.CalledProcessError):
        return {"available": False}


def timed(function, device, iterations):
    for _ in range(3):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = []
    for _ in range(iterations):
        start = time.perf_counter()
        function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed.append((time.perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(elapsed)),
            "p95_ms": float(np.percentile(elapsed, 95)),
            "iterations": iterations, "warmup_iterations": 3}


def dataset_examples(manifest):
    images, records = [], []
    for task in manifest["tasks"]:
        dataset = AlignedLiberoSequenceDataset(DEFAULT_MANIFEST, task["task_id"], "train")
        try:
            with h5py.File(dataset.hdf5_path, "r", swmr=True, libver="latest") as file:
                for episode in dataset.demos[:2]:
                    length = len(file[f"data/{episode}/actions"])
                    for row in (0, (length - 2) // 2, length - 2):
                        sample = dataset[dataset.index_for(episode, row)]
                        raw = file[f"data/{episode}/obs/agentview_rgb"][row]
                        # Independent raw/live boundary versus the existing adapter.
                        torch.testing.assert_close(
                            canonical_live_rgb(raw), torch.from_numpy(sample["image"]),
                            rtol=0, atol=0)
                        images.append(torch.from_numpy(sample["image"]))
                        records.append({**sample["metadata"], "instruction": sample["instruction"]})
        finally:
            dataset.close()
    return torch.stack(images), records


def verify_preprocessing(encoder, images):
    pixels = encoder.preprocess_images(images)
    # Independent PIL bicubic resize and channel normalization, no center crop.
    reference = []
    for image in images:
        rgb = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        resized = np.asarray(Image.fromarray(rgb).resize((224, 224), Image.Resampling.BICUBIC))
        normalized = resized.astype(np.float32) / 255
        normalized = (normalized - np.asarray(encoder.processor.image_mean, dtype=np.float32))
        normalized /= np.asarray(encoder.processor.image_std, dtype=np.float32)
        reference.append(torch.from_numpy(normalized.transpose(2, 0, 1).copy()))
    reference = torch.stack(reference)
    torch.testing.assert_close(pixels, reference, rtol=0, atol=1e-6)
    for invalid in (images.mul(255), images.to(torch.uint8), images[:, :, :64],
                    torch.full_like(images, float("nan")), images[:0]):
        rejects(lambda: encoder.preprocess_images(invalid))
    return pixels, {"pil_reference_max_abs_error": (pixels - reference).abs().max().item(),
                    "invalid_inputs_rejected": 5, "pixel_values": stats(pixels)}


def verify_text(encoder, instructions):
    tokens = encoder.tokenize(instructions)
    fresh, mask = encoder.encode_text(instructions, use_cache=False)
    cached, cached_mask = encoder.encode_text(instructions)
    repeated, _ = encoder.encode_text(instructions)
    torch.testing.assert_close(fresh, cached, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(cached, repeated, rtol=0, atol=0)
    check(torch.equal(mask, cached_mask), "Cache changed text mask")
    check(torch.equal(mask.cpu(), tokens["attention_mask"].bool()), "Invalid text mask")
    check(fresh.shape == (len(instructions), 77, 512), "Wrong text shape")
    rows = []
    for i, instruction in enumerate(instructions):
        length = len(encoder.tokenizer(instruction, truncation=False)["input_ids"])
        check(mask[i].sum().item() == length, "Wrong token count")
        check(mask[i, :length].all() and not mask[i, length:].any(), "Non-contiguous validity")
        check(tokens["input_ids"][i, 0] == encoder.tokenizer.bos_token_id, "Missing BOS")
        check(tokens["input_ids"][i, length - 1] == encoder.tokenizer.eos_token_id, "Missing EOS")
        rows.append({"instruction": instruction, "valid_tokens": length,
                     "tokens": encoder.tokenizer.convert_ids_to_tokens(tokens["input_ids"][i, :length].tolist()),
                     "ids": tokens["input_ids"][i, :length].tolist()})
    boundary = encoder.tokenize(["bowl " * 75])
    check(boundary["attention_mask"].sum().item() == 77, "77-token boundary failed")
    rejects(lambda: encoder.tokenize(["bowl " * 76]))
    rejects(lambda: encoder.tokenize("a string instead of a batch"))
    blank, blank_mask = encoder.encode_text([""])
    check(blank_mask.sum().item() == 2, "Blank instruction must retain BOS/EOS")
    check(not torch.allclose(fresh[0], fresh[1]), "Different instructions have identical features")
    # Reordering, duplicate instructions and caller mutation must not corrupt cache.
    reordered, _ = encoder.encode_text([instructions[1], instructions[0], instructions[1]])
    torch.testing.assert_close(reordered[0], cached[1], rtol=0, atol=0)
    torch.testing.assert_close(reordered[1], cached[0], rtol=0, atol=0)
    repeated.zero_()
    intact, _ = encoder.encode_text(instructions)
    torch.testing.assert_close(intact, cached, rtol=0, atol=0)
    return {"features": stats(fresh), "instructions": rows, "mask_shape": list(mask.shape),
            "cache_max_abs_error": (fresh - cached).abs().max().item(),
            "cache_reorder_duplicates_and_mutation": "PASS", "blank_valid_tokens": 2,
            "77_tokens_accepted_78_rejected": True}


def save_patch_visual(image, pixels, features):
    """Raw view, actual encoder view with grid, and patch feature norms.

    Feature norm is only a numerical diagnostic, not attention or an object map.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb = image.permute(1, 2, 0).numpy()
    # Undo only checkpoint normalization for a human-facing view.
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    view = np.clip(pixels.permute(1, 2, 0).numpy() * std + mean, 0, 1)
    grid = Image.fromarray((view * 255).round().astype(np.uint8))
    draw = ImageDraw.Draw(grid)
    for edge in range(16, 224, 16):
        draw.line((edge, 0, edge, 223), fill=(255, 195, 0), width=1)
        draw.line((0, edge, 223, edge), fill=(255, 195, 0), width=1)
    norms = features.float().norm(dim=-1).cpu().reshape(14, 14).numpy()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), constrained_layout=True)
    axes[0].imshow(rgb); axes[0].set_title("Upright dataset RGB: 128 x 128")
    axes[1].imshow(grid); axes[1].set_title("CLIP input: 14 x 14 patch grid")
    im = axes[2].imshow(norms, cmap="viridis")
    axes[2].set_title("Final patch feature L2 norm\nDiagnostic, not object localization")
    fig.colorbar(im, ax=axes[2], shrink=0.8)
    for axis in axes:
        axis.axis("off")
    fig.savefig(OUTPUT / "patch_inspection.png", dpi=160)
    plt.close(fig)


def verify_live(encoder, manifest):
    from verify_libero_replay_alignment import create_environment, reset_episode, restore_state
    import robosuite.macros as macros
    check(macros.IMAGE_CONVENTION == "opengl", "Live image convention changed")
    reports = []
    for task in manifest["tasks"]:
        with h5py.File(ROOT / task["file"], "r") as file:
            env = create_environment(file["data"])
            try:
                env.seed(20261007)
                observation = env.reset()
                fresh = canonical_live_rgb(observation["agentview_image"])
                fresh_tokens = encoder.encode_images(fresh.unsqueeze(0))
                name = task["splits"]["train"]["episodes"][0]
                episode = file[f"data/{name}"]
                row = (len(episode["actions"]) - 2) // 2
                reset_episode(env, episode)
                observation = restore_state(env, episode["states"][row + 1])
                raw_live = observation["agentview_image"]
                raw_stored = episode["obs/agentview_rgb"][row]
                live = canonical_live_rgb(raw_live)
                recorded = canonical_live_rgb(raw_stored)
                correct_mae = float((live - recorded).abs().mean() * 255)
                flipped_mae = float((live.flip(-2) - recorded).abs().mean() * 255)
                check(correct_mae < flipped_mae, "Live/stored image orientation disagrees")
                pair = encoder.encode_images(torch.stack([recorded, live]))
                # Exercise a new frame after an existing demonstration command.
                new_obs, _, _, _ = env.step(episode["actions"][row + 1])
                new_tokens = encoder.encode_images(canonical_live_rgb(new_obs["agentview_image"])[None])
                montage = np.concatenate([
                    (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                    for x in (fresh, recorded, live)], axis=1)
                Image.fromarray(montage).save(OUTPUT / f"{task['task_id']}_live_upright.png")
                reports.append({"task_id": task["task_id"], "episode": name, "row": row,
                                "fresh_reset_features": stats(fresh_tokens),
                                "recorded_and_restored_features": stats(pair),
                                "post_step_features": stats(new_tokens),
                                "correct_orientation_pixel_mae": correct_mae,
                                "wrong_orientation_pixel_mae": flipped_mae,
                                "gpu_with_renderer": gpu_snapshot()})
            finally:
                env.close()
    return reports


def inspect(args):
    torch.set_num_threads(4)
    torch.manual_seed(20261007)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    check(args.device != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    provenance = json.loads((OUTPUT / "checkpoint_provenance.json").read_text())
    check(provenance["revision"] == REVISION, "Wrong checkpoint revision")
    check(sha256(MODEL_DIR / "pytorch_model.bin") == WEIGHTS_SHA256, "Weight integrity failed")
    for name, expected in provenance["file_sha256"].items():
        if name != "pytorch_model.bin":
            check(sha256(MODEL_DIR / name) == expected, f"Checkpoint resource changed: {name}")
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    images, records = dataset_examples(manifest)
    print(f"Inspecting {len(images)} actual training frames from all three tasks", flush=True)
    encoder = FrozenCLIPEncoder(args.device)
    encoder.train(True)
    check(all(not module.training for module in encoder.modules()), "Frozen module entered train mode")
    check(all(not p.requires_grad for p in encoder.parameters()), "Trainable encoder parameter")
    pixels, preprocessing = verify_preprocessing(encoder, images)
    with torch.no_grad():
        full = encoder.vision(pixel_values=pixels[:1].to(encoder.device)).last_hidden_state
    check(full.shape == (1, 197, 768), "Wrong full visual token shape")
    visual = encoder.encode_images(images)
    check(visual.shape == (len(images), 196, 768), "Wrong patch shape")
    single = encoder.encode_images(images[:1])
    torch.testing.assert_close(single, full[:, 1:], rtol=0, atol=0)
    repeated = encoder.encode_images(images[:1])
    torch.testing.assert_close(single, repeated, rtol=1e-5, atol=1e-5)
    # CUDA can select different accumulation kernels for B=1 and B=18.
    # This is distinct from repeating an identical input AND batch shape.
    torch.testing.assert_close(single[0], visual[0], rtol=1e-4, atol=1e-3)
    check(not single.requires_grad and single.grad_fn is None and not single.is_inference(),
          "Encoder must return ordinary detached tensors")
    text_report = verify_text(encoder, [task["instruction"] for task in manifest["tasks"]])
    outputs = encoder(images[:3], [record["instruction"] for record in records[:3]])
    check(all(not value.requires_grad for value in outputs.values()), "Output has gradients")
    check(all(p.grad is None for p in encoder.parameters()), "Encoder gradient found")
    rejects(lambda: encoder(images[:2], ["one instruction"]))
    save_patch_visual(images[1], pixels[1], visual[1])
    print("Shapes, normalization, masks, cache, freezing and repeatability: PASS", flush=True)
    live = verify_live(encoder, manifest)
    print("Live RGB and feature extraction for all three tasks: PASS", flush=True)
    profiles = []
    for batch_size in (1, args.batch_size):
        batch = images[:batch_size]
        check(len(batch) == batch_size, "Requested profile batch exceeds inspected samples")
        prepared = encoder.preprocess_images(batch).to(encoder.device)
        if encoder.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(encoder.device)
        profile = {"batch_size": batch_size,
                   "preprocess": timed(lambda: encoder.preprocess_images(batch), encoder.device, args.iterations),
                   "vision_on_device": timed(lambda: encoder.encode_pixels(prepared), encoder.device, args.iterations),
                   "image_end_to_end": timed(lambda: encoder.encode_images(batch), encoder.device, args.iterations)}
        if encoder.device.type == "cuda":
            profile.update(peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                           peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                           gpu=gpu_snapshot())
            check(profile["peak_reserved_mib"] < 5000, "Encoder exceeds VRAM headroom target")
        profiles.append(profile)
    instruction = [manifest["tasks"][0]["instruction"]]
    text_timing = {"uncached": timed(lambda: encoder.encode_text(instruction, use_cache=False),
                                    encoder.device, args.iterations),
                   "cached": timed(lambda: encoder.encode_text(instruction), encoder.device, args.iterations)}
    return {"status": "PASS", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": 3, "checkpoint": CHECKPOINT, "revision": REVISION,
            "checkpoint_weights_sha256": WEIGHTS_SHA256,
            "preprocessing_version": PREPROCESSING_VERSION,
            "manifest_sha256": sha256(DEFAULT_MANIFEST), "seed": 20261007,
            "device": str(encoder.device), "weight_dtype": str(next(encoder.parameters()).dtype),
            "attention": "PyTorch SDPA", "torch_cpu_threads": torch.get_num_threads(),
            "versions": {name: importlib.metadata.version(name) for name in
                         ["torch", "transformers", "huggingface-hub", "tokenizers", "safetensors",
                          "numpy", "Pillow", "h5py", "robomimic", "mujoco", "robosuite"]},
            "encoder_parameters": sum(p.numel() for p in encoder.parameters()),
            "trainable_parameters": 0, "parameters_with_gradients": 0,
            "frozen_even_after_train_call": True, "outputs_are_ordinary_detached_tensors": True,
            "samples": records, "preprocessing": preprocessing,
            "full_visual_shape": list(full.shape), "patch_features": stats(visual),
            "repeat_max_abs_error": (single - repeated).abs().max().item(),
            "single_vs_batch_max_abs_error": (single[0] - visual[0]).abs().max().item(),
            "single_vs_batch_tolerance": {"rtol": 1e-4, "atol": 1e-3},
            "text": text_report, "live": live, "performance": profiles, "text_timing": text_timing,
            "limitations": ["No action policy, training, or manipulation competence evaluated.",
                            "Latency is local FP32 encoder timing, not full control-loop throughput.",
                            "PyTorch memory excludes renderer/driver allocations; GPU snapshots include other apps."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.iterations < 1 or not 1 <= args.batch_size <= 18:
        parser.error("iterations must be positive and batch-size must be 1..18")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT / "verification.json"
    report_path.write_text('{"status": "RUNNING"}\n')
    try:
        report = inspect(args)
    except Exception as error:
        report_path.write_text(json.dumps({"status": "FAIL", "error": str(error)}, indent=2) + "\n")
        raise
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Stage 3 PASS. Report:", report_path, flush=True)
    print(json.dumps(report["performance"], indent=2), flush=True)


if __name__ == "__main__":
    main()
