"""Download the remaining original Goal files at the audited immutable revision."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

from huggingface_hub import HfApi, hf_hub_download

REPO = "yifengzhu-hf/LIBERO-datasets"
REVISION = "97773100c1474cd0d686ebd173cc0e4fd5442466"


def main():
    info = HfApi(token=False).dataset_info(REPO, revision=REVISION, files_metadata=True)
    files = [f for f in info.siblings if f.rfilename.startswith("libero_goal/")
             and f.rfilename.endswith(".hdf5")]
    assert len(files) == 10, [f.rfilename for f in files]
    records = []
    for file in files:
        path = ROOT / "data" / "datasets" / file.rfilename
        if not path.exists():
            print("Downloading", file.rfilename, flush=True)
            hf_hub_download(REPO, file.rfilename, repo_type="dataset", revision=REVISION,
                            token=False, local_dir=ROOT / "data" / "datasets")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        assert path.stat().st_size == file.size
        assert digest == file.lfs.sha256, (path, digest)
        records.append(dict(file=file.rfilename, bytes=file.size, sha256=digest))
        print("Verified", file.rfilename, flush=True)
    output = ROOT / "outputs" / "stage6" / "download_provenance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(repo=REPO, revision=REVISION, files=records), indent=2)+"\n")


if __name__ == "__main__":
    main()
