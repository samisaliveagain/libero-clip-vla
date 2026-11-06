# Setup and reproduction

The original runs used native Windows, Python 3.11.11, PyTorch 2.6.0+cu126, an RTX 4050 Laptop GPU (6 GB), LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`, robosuite 1.4.0, and MuJoCo 2.3.7. Other platforms have not been verified by this project. These are historical, intentionally pinned experiment dependencies.

Commands below use Windows CMD and run from this repository's root. If you already have the verified environment, activate it instead of reinstalling it.

## Environment

```cmd
git submodule update --init
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python scripts\configure_libero.py --windows-fixes
python scripts\configure_libero.py --check
```

The configuration helper writes clone-relative absolute paths into the ignored `.libero/config.yaml`. Project entry points expose the pinned LIBERO checkout's nested Python layout directly, so no unpinned `pip install libero` is needed.

The optional `--windows-fixes` flag applies the three documented compatibility steps to the active virtual environment: creates `robosuite/macros_private.py` if absent, copies the installed MuJoCo DLL into robosuite's utility directory, and selects WGL on Windows in the pinned renderer. It rejects unexpected renderer source rather than patching it blindly. These steps retain the simulator/controller versions used by the experiment.

## Public data and frozen encoder

```cmd
python scripts\download_project_artifacts.py --dataset --checkpoint
python scripts\download_clip_stage3.py
```

The artifact downloader restores the unchanged ten-file Goal subset from the project's Hugging Face dataset mirror, plus the selected frozen policy. Hub revisions and SHA-256 values are pinned in `docs/artifacts.json`. The original data source is revision `97773100c1474cd0d686ebd173cc0e4fd5442466` of `yifengzhu-hf/LIBERO-datasets`; `scripts/download_libero_goal.py` remains an upstream-download alternative. Expect approximately 6.4 GB for demonstrations and 0.6 GB for CLIP, plus environment and output storage. The checked-in `data/manifests/` preserves the audited splits and normalizers; do not regenerate it casually when comparing with the published results.

## Inspect and train

```cmd
python scripts\inspect_libero_env.py
python scripts\libero_dataset_adapter.py
python scripts\inspect_libero_policy.py --batch-size 2 --output outputs\verification\policy
python scripts\train_libero_policy.py --mode train --tasks all --manifest data\manifests\libero_goal_stage6.json --output outputs\reproduction --epochs 60 --patience 15 --microbatch 32 --accumulation 1 --workers 1 --validation-workers 0 --seed 20261008 --cpu-threads 4
```

Lower the microbatch and increase accumulation if memory is constrained; changing these settings can change the training trajectory. New training is not expected to reproduce exact saved weights bit for bit. Never select a checkpoint using final-reset results.

## Reproduce the saved policy's rollout protocol

The exact frozen policy is a separate 42.8 MB artifact on [Hugging Face](https://huggingface.co/Samisaliveagain/libero-clip-vla), restored by `--checkpoint` to `outputs/stage7/frozen_epoch06.ckpt`. The downloader verifies SHA-256 `3731f123462be5bee742cfcb45a595386b68a5963150e1a7079328315eca36cf`. It is not included in the GitHub source repository.

```cmd
python scripts\rollout_libero_policy.py --checkpoint outputs\stage7\frozen_epoch06.ckpt --manifest data\manifests\libero_goal_stage6.json --task put_the_bowl_on_top_of_the_cabinet --reset-indices 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 --settling-steps 10 --settling-gripper -1 --horizon 600 --output outputs\reproduction\cabinet
```

Use a new output directory for each invocation; the runner refuses to overwrite an existing report. The ten-open protocol is intentionally different from the earlier five-zero development protocol. Keep their results separate. The saved reports can be inspected without downloading data or running simulation.

Publication preparation checked the current environment and source package; a completely fresh environment installation has not been performed as part of that cleanup.
