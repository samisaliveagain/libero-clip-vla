"""Download the selected public checkpoint, pinned and hash-verified.

NON-LEARNED INFRASTRUCTURE. local_dir avoids Windows symlink privileges.
No Hugging Face account or remote model code is needed.
"""

import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from frozen_clip_encoder import CHECKPOINT, MODEL_DIR, REVISION, ROOT

WEIGHTS_SHA256 = "ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0"
FILES = ["config.json", "pytorch_model.bin", "preprocessor_config.json",
         "merges.txt", "vocab.json", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "README.md"]


def main():
    path = Path(snapshot_download(
        CHECKPOINT, revision=REVISION, token=False,
        local_dir=MODEL_DIR, allow_patterns=FILES,
    ))
    hashes = {}
    for name in FILES:
        with (path / name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    if hashes["pytorch_model.bin"] != WEIGHTS_SHA256:
        raise RuntimeError("Checkpoint SHA-256 does not match its upstream LFS hash.")
    report = {"checkpoint": CHECKPOINT, "revision": REVISION,
              "weights_sha256": WEIGHTS_SHA256, "file_sha256": hashes,
              "weights_bytes": (path / "pytorch_model.bin").stat().st_size,
              "local_snapshot": str(path)}
    output = ROOT / "outputs" / "stage3"
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoint_provenance.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Verified checkpoint:", path, flush=True)


if __name__ == "__main__":
    main()
