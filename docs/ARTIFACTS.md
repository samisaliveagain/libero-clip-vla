# Downloadable data and experiment artifacts

The source repository stays small. Heavy files are versioned separately on Hugging Face, and the public manifests record their SHA-256 hashes.

| Resource | Location | Local restore path |
| --- | --- | --- |
| Ten original Goal HDF5 files (6.37 GB) | [Dataset repository](https://huggingface.co/datasets/Samisaliveagain/libero-goal-vla) | `data/datasets/libero_goal/` |
| Selected frozen policy (42.8 MB) | [Model repository](https://huggingface.co/Samisaliveagain/libero-clip-vla/blob/main/outputs/stage7/frozen_epoch06.ckpt) | `outputs/stage7/frozen_epoch06.ckpt` |
| All saved checkpoints, raw videos/traces, configs and logs (~1.00 GB) | [Experiment archive](https://huggingface.co/Samisaliveagain/libero-clip-vla/tree/main/outputs) | Original `outputs/stage2` through `outputs/stage7` paths |
| Frozen CLIP backbone (~0.60 GB) | [Upstream model](https://huggingface.co/openai/clip-vit-base-patch16) | `data/models/clip-vit-base-patch16/` |

## Restore only what you need

From the project root in Windows CMD, using the configured environment:

```cmd
python scripts\download_project_artifacts.py --dataset --checkpoint
python scripts\download_clip_stage3.py
```

To restore every saved training checkpoint and raw experiment artifact:

```cmd
python scripts\download_project_artifacts.py --all-artifacts
```

The downloader restores files directly to their expected project paths and verifies their SHA-256 values. Existing matching files are reused; existing files with conflicting content cause an error instead of being overwritten. The Hub revisions are pinned in `docs/artifacts.json`, so later Hub updates do not silently change a reproduction.

The selected checkpoint hash is `3731f123462be5bee742cfcb45a595386b68a5963150e1a7079328315eca36cf`. It is a project-produced Lightning checkpoint; use the project's custom loader rather than `AutoModel`. These checkpoints use PyTorch serialization and should only be loaded from a trusted source.

The dataset remains the original LIBERO-Goal release, not newly collected data. Source revision, file hashes, audited temporal alignment, train/validation/test episode IDs, and training-only normalizers are preserved. See the dataset card for upstream attribution and licensing.
