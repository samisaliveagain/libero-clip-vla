"""Measure held-out action error and verify exact policy serialization.

This is offline evaluation, not evidence of successful robot control.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from clip_features import FrozenCLIPFeatures
from libero_dataset_adapter import AlignedLiberoSequenceDataset, DEFAULT_MANIFEST
from libero_policy import LiberoActionPolicy


def checkpoint_policy(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    policy = LiberoActionPolicy().to(device).eval()
    policy.load_state_dict({k.removeprefix("policy."): v for k, v in
                           checkpoint["state_dict"].items() if k.startswith("policy.")})
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tasks", nargs="+", default=["put_the_bowl_on_the_plate"])
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    features = FrozenCLIPFeatures()
    policy = checkpoint_policy(args.checkpoint, features.device)
    buffer = io.BytesIO()
    torch.save(policy.state_dict(), buffer)
    buffer.seek(0)
    restored = LiberoActionPolicy().to(features.device).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    results = []
    reload_error = 0.0
    with torch.inference_mode():
        for task in args.tasks:
            data = AlignedLiberoSequenceDataset(args.manifest, task, args.split)
            squared = torch.zeros(7, device=features.device)
            absolute = torch.zeros_like(squared)
            sign_correct = sign_count = saturated = count = 0
            try:
                for i, batch in enumerate(DataLoader(data, batch_size=8, num_workers=0)):
                    visual = features.encode_images(batch["image"])
                    text, mask = features.encode_text(batch["instruction"])
                    proprio = batch["proprio"].to(features.device)
                    target = batch["action"].to(features.device)
                    prediction = policy(visual, text, mask, proprio)
                    if i == 0:
                        reloaded = restored(visual, text, mask, proprio)
                        reload_error = max(reload_error, (prediction - reloaded).abs().max().item())
                    error = prediction - target
                    squared += error.square().sum(0)
                    absolute += error.abs().sum(0)
                    nonzero = target[:, 6].abs() > 1e-6
                    sign_correct += ((prediction[:, 6].sign() == target[:, 6].sign()) & nonzero).sum().item()
                    sign_count += nonzero.sum().item()
                    saturated += (prediction.abs() > .99).sum().item()
                    count += len(target)
            finally:
                data.close()
            mse = (squared / count).cpu().tolist()
            result = dict(task=task, samples=count, mse_per_dimension=mse,
                          mae_per_dimension=(absolute / count).cpu().tolist(),
                          translation_mse=sum(mse[:3])/3, rotation_mse=sum(mse[3:6])/3,
                          gripper_mse=mse[6], loss=sum(mse[:6])/6+mse[6],
                          gripper_sign_agreement=sign_correct/sign_count,
                          nonzero_gripper_targets=sign_count,
                          saturation_fraction=saturated/(count*7))
            results.append(result)
            print(json.dumps(result), flush=True)
    assert reload_error == 0.0, reload_error
    report = dict(checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  manifest=str(args.manifest.resolve()), split=args.split,
                  reload_max_absolute_difference=reload_error, results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
