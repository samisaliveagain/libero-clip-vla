# Publication layout

| Repository | What belongs there |
| --- | --- |
| [GitHub: libero-clip-vla](https://github.com/samisaliveagain/libero-clip-vla) | Code, setup, architecture, compact results, manifests, six curated videos, GIF preview |
| [Hugging Face: policy artifacts](https://huggingface.co/Samisaliveagain/libero-clip-vla) | All saved checkpoints and raw experiment outputs, model card, MIT license, SHA-256 inventory |
| [Hugging Face: dataset](https://huggingface.co/datasets/Samisaliveagain/libero-goal-vla) | Unchanged ten-file LIBERO-Goal subset, audited splits, original Apache-2.0 licensing and attribution |

Original code and trained policy checkpoints are MIT licensed, as selected by the project author. Third-party data, assets, and pretrained models retain their original terms. ChatGPT instructions, personal interview material, caches, local datasets/models, and raw runs are Git-ignored.

## Reproduce from the published source

Commands use Windows CMD:

```cmd
git clone --recurse-submodules https://github.com/samisaliveagain/libero-clip-vla.git
cd libero-clip-vla
```

Follow [SETUP.md](SETUP.md), then restore verified data and the selected checkpoint with:

```cmd
python scripts\download_project_artifacts.py --dataset --checkpoint
python scripts\download_clip_stage3.py
```

## Future updates

Keep large files on Hugging Face. After intentionally updating Hub artifacts, update the pinned revisions and hash inventory in `docs/artifacts.json`, verify them, and commit that small manifest to GitHub. A later Hub change must not silently replace the artifacts used to reproduce these results.

Review `git diff --cached --stat` before pushing source updates. Keep the per-task results and evaluation caveats in sync with saved reports; curated success clips are demonstrations of selected outcomes, not an aggregate success claim.
