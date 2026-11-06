"""Stage 2: inspect original HDF5 read-only; export an episode and audit JSON.

NON-LEARNED INFRASTRUCTURE. No model imports, fitting, or alignment assumptions.
Run from Windows CMD: venv\Scripts\python.exe scripts\inspect_libero_dataset.py
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import cv2
import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

DEFAULT = ROOT / "data/datasets/libero_goal/put_the_bowl_on_the_plate_demo.hdf5"


def json_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def upright_rgb(images: np.ndarray) -> np.ndarray:
    """Convert LIBERO's bottom-left OpenGL rows to normal top-left display."""
    height_axis = images.ndim - 3
    return np.flip(images, axis=height_axis).copy()


def export_episode(episode, output, frequency):
    """Export stored rows; the verified training target is action[k+1]."""
    name = episode.name.rsplit("/", 1)[-1]
    source_images = episode["obs/agentview_rgb"][:]
    images = upright_rgb(source_images)
    actions = episode["actions"][:]
    n = len(actions)
    video = output / f"{name}_recorded_upright.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), frequency, (128, 128))
    if not writer.isOpened():
        raise RuntimeError("Could not open MP4 writer")
    try:
        for image in images:
            writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    # Verify the encoded video includes every recorded image.
    cap = cv2.VideoCapture(str(video))
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    assert count == n, (count, n)

    indices = np.linspace(0, n - 1, 6, dtype=int)
    sheet = Image.new("RGB", (3 * 256, 2 * 286), "white")
    draw = ImageDraw.Draw(sheet)
    for cell, row in enumerate(indices):
        x, y = (cell % 3) * 256, (cell // 3) * 286
        sheet.paste(Image.fromarray(images[row]).resize((256, 256)), (x, y))
        draw.text((x + 8, y + 260), f"{name} | stored row {row}", fill="black")
    sheet.save(output / f"{name}_frames_upright.png")

    if name == "demo_0":
        # Embed a small, lossy preview; the source HDF5 remains untouched.
        preview = {"actions": actions.round(6).tolist(),
                   "fingers": episode["obs/gripper_states"][:].round(6).tolist(),
                   "image_convention": "upright (vertical flip of source OpenGL rows)",
                   "images": []}
        for frame in images:
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, 65])
            assert ok
            preview["images"].append("data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"))
        template = (ROOT / "scripts/stage2_episode_visual.html").read_text(encoding="utf-8")
        fragment = template.replace("__EPISODE_DATA__", json.dumps(preview, separators=(",", ":")))
        assert len(fragment.encode("utf-8")) < 1_000_000
        (output / "recorded-episode.html").write_text(fragment, encoding="utf-8")

    fig, axes = plt.subplots(5, 1, figsize=(11, 11), sharex=True, constrained_layout=True)
    panels = [
        (actions[:, :3], ["dx", "dy", "dz"], "Translation command\n(normalized)"),
        (actions[:, 3:6], ["dRx", "dRy", "dRz"], "Rotation command\n(normalized)"),
        (actions[:, 6:], ["gripper command"], "Gripper command\n(-1 open, +1 close)"),
        (episode["obs/gripper_states"][:], ["finger 1", "finger 2"], "Measured finger\nposition (m)"),
        (episode["obs/joint_states"][:], [f"joint {j+1}" for j in range(7)], "Measured arm\nposition (rad)"),
    ]
    for ax, (values, labels, ylabel) in zip(axes, panels):
        ax.plot(values, linewidth=1.25)
        ax.set_ylabel(ylabel)
        ax.legend(labels, loc="upper right", ncol=min(len(labels), 4), fontsize=8)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Stored row k (observation[k] was captured after action[k])")
    fig.suptitle(f"{name}: recorded commands and measured gripper state")
    fig.savefig(output / f"{name}_traces.png", dpi=145)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=DEFAULT)
    parser.add_argument("--demo", default="demo_0")
    args = parser.parse_args()
    output = ROOT / "outputs/stage2" / args.file.stem
    output.mkdir(parents=True, exist_ok=True)
    verified_alignment = {
        "put_the_bowl_on_the_plate_demo.hdf5": {
            "status": "VERIFIED_POST_ACTION",
            "training_pair": "obs[k] -> action[k+1]",
            "valid_k": "0..T-2",
        },
        "put_the_bowl_on_the_stove_demo.hdf5": {
            "status": "VERIFIED_POST_ACTION",
            "training_pair": "obs[k] -> action[k+1]",
            "valid_k": "0..T-2",
        },
        "put_the_bowl_on_top_of_the_cabinet_demo.hdf5": {
            "status": "VERIFIED_POST_ACTION",
            "training_pair": "obs[k] -> action[k+1]",
            "valid_k": "0..T-2",
        },
    }
    report = {"file": str(args.file.resolve()), "bytes": args.file.stat().st_size,
              "sha256": file_hash(args.file),
              "source_image_convention": "opengl (bottom-left row origin)",
              "inspection_image_convention": "upright (top-left row origin)",
              "model_preprocessing": "vertical flip exactly once before image encoding",
              "alignment": verified_alignment.get(args.file.name, {"status": "PENDING_REPLAY_AUDIT"})}
    with h5py.File(args.file, "r") as f:
        data = f["data"]
        report["metadata"] = {k: json_value(v) for k, v in data.attrs.items()}
        env_args = json.loads(data.attrs["env_args"])
        frequency = env_args["env_kwargs"]["control_freq"]
        report["frequency_hz_metadata"] = frequency
        report["schema"] = {}
        data[args.demo].visititems(lambda name, obj: report["schema"].update(
            {name: {"shape": list(obj.shape), "dtype": str(obj.dtype)}}
        ) if isinstance(obj, h5py.Dataset) else None)
        report["episodes"] = []
        fingerprints = {}
        all_actions = []
        for name in sorted(data, key=lambda s: int(s.split("_")[-1])):
            ep = data[name]
            a = ep["actions"][:]
            n = len(a)
            assert a.shape == (n, 7) and n > 1
            assert ep.attrs["num_samples"] == n
            for key, shape in (("joint_states", (n, 7)), ("gripper_states", (n, 2)),
                               ("agentview_rgb", (n, 128, 128, 3))):
                assert ep[f"obs/{key}"].shape == shape
            for key in ("actions", "states", "obs/joint_states", "obs/gripper_states"):
                assert np.isfinite(ep[key][:]).all(), (name, key)
            assert ep["obs/agentview_rgb"].dtype == np.uint8
            digest = hashlib.sha256(a.tobytes() + ep["states"][:].tobytes()).hexdigest()
            fingerprints.setdefault(digest, []).append(name)
            report["episodes"].append({"name": name, "rows": n, "trajectory_sha256": digest,
                "dones_indices": np.flatnonzero(ep["dones"][:]).tolist(),
                "rewards_indices": np.flatnonzero(ep["rewards"][:]).tolist(),
                "state_time_first": float(ep["states"][0, 0]),
                "state_time_last": float(ep["states"][-1, 0]),
                "max_state_time_step_error_s": float(np.max(np.abs(np.diff(ep["states"][:, 0]) - 1/frequency)))})
            all_actions.append(a)
        actions = np.concatenate(all_actions)
        report.update(episode_count=len(all_actions), row_count=len(actions),
                      action_min=actions.min(axis=0).tolist(), action_max=actions.max(axis=0).tolist(),
                      out_of_bounds_values=int(np.sum(np.abs(actions) > 1 + 1e-6)),
                      exact_duplicate_groups=[v for v in fingerprints.values() if len(v) > 1])
        export_episode(data[args.demo], output, frequency)
    (output / "inspection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("episodes", "metadata")}, indent=2))
    print("Output:", output)


if __name__ == "__main__":
    main()
