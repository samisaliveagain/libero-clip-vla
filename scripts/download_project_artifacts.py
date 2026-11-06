"""Restore hash-verified project artifacts from pinned Hugging Face revisions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restore(repo: dict, files: list[dict]) -> None:
    from huggingface_hub import hf_hub_download

    for item in files:
        destination = (ROOT / item["local_path"]).resolve()
        if not destination.is_relative_to(ROOT):
            raise ValueError("Artifact path leaves the project directory")
        if destination.exists():
            if destination.stat().st_size != item["bytes"] or sha256(destination) != item["sha256"]:
                raise FileExistsError(f"Conflicting local file; preserve or move it first: {destination}")
            print("Verified existing", item["local_path"], flush=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Avoid keeping a second full download in the global Hub blob cache.
        with tempfile.TemporaryDirectory(prefix=".hf-restore-", dir=destination.parent) as temporary:
            downloaded = Path(hf_hub_download(
                repo_id=repo["repo_id"], repo_type=repo["repo_type"], revision=repo["revision"],
                filename=item["path"], local_dir=temporary,
            ))
            if downloaded.stat().st_size != item["bytes"] or sha256(downloaded) != item["sha256"]:
                raise RuntimeError(f"Artifact hash mismatch: {item['path']}")
            if destination.exists():
                raise FileExistsError(f"Destination appeared during download: {destination}")
            os.replace(downloaded, destination)
        print("Restored", item["local_path"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--all-artifacts", action="store_true")
    args = parser.parse_args()
    if not any((args.dataset, args.checkpoint, args.all_artifacts)):
        parser.error("Choose --dataset, --checkpoint, and/or --all-artifacts.")
    metadata = json.loads((ROOT / "docs/artifacts.json").read_text(encoding="utf-8"))
    if args.dataset:
        restore(metadata["dataset"], metadata["dataset"]["files"])
    if args.all_artifacts:
        restore(metadata["model"], metadata["model"]["files"])
    elif args.checkpoint:
        restore(metadata["model"], [f for f in metadata["model"]["files"] if f["path"] == metadata["selected_checkpoint"]])


if __name__ == "__main__":
    main()
