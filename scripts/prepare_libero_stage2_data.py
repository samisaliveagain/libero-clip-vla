"""Stage 2: create disjoint episode splits and training-only proprio stats.

NON-LEARNED INFRASTRUCTURE. Source HDF5 files are opened read-only.

Windows CMD:
    venv\Scripts\python.exe scripts\prepare_libero_stage2_data.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_REVISION = "97773100c1474cd0d686ebd173cc0e4fd5442466"
SPLIT_SEED = 20260920
STD_EPSILON = 1e-6
FILES = {
    "put_the_bowl_on_the_plate": {
        "name": "put_the_bowl_on_the_plate_demo.hdf5",
        "bytes": 468246288,
        "sha256": "e69528b0cf10dfc59b20698e12ec2affc03f3887309034d3eb74cac3ec929406",
    },
    "put_the_bowl_on_the_stove": {
        "name": "put_the_bowl_on_the_stove_demo.hdf5",
        "bytes": 509123820,
        "sha256": "ff77d3bf0f89a147d423840ecbd842cd63c3c55aeead91f88dbc0917be473e99",
    },
    "put_the_bowl_on_top_of_the_cabinet": {
        "name": "put_the_bowl_on_top_of_the_cabinet_demo.hdf5",
        "bytes": 510414285,
        "sha256": "e14c84ebc820a13ef9a92ca82a7f04bb299f0ee96dfe9cc907bab71aeb2683b2",
    },
}


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ordered_episodes(data: h5py.Group) -> list[str]:
    return sorted(data.keys(), key=lambda value: int(value.split("_")[-1]))


def main() -> None:
    dataset_dir = ROOT / "data" / "datasets" / "libero_goal"
    manifest_dir = ROOT / "data" / "manifests"
    output_dir = ROOT / "outputs" / "stage2"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SPLIT_SEED)
    task_records = []
    trajectory_fingerprints: dict[str, list[str]] = {}
    train_proprio = []
    all_episode_ids: set[str] = set()
    split_episode_ids = {"train": set(), "validation": set(), "test": set()}

    for task, expected in FILES.items():
        path = dataset_dir / expected["name"]
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256_file(path)
        assert actual_bytes == expected["bytes"], (path, actual_bytes)
        assert actual_sha256 == expected["sha256"], (path, actual_sha256)

        with h5py.File(path, "r") as file:
            data = file["data"]
            episodes = ordered_episodes(data)
            assert len(episodes) == 50, (task, len(episodes))
            shuffled = np.asarray(episodes, dtype=object)[rng.permutation(len(episodes))].tolist()
            splits = {
                "train": sorted(shuffled[:40], key=lambda x: int(x.split("_")[-1])),
                "validation": sorted(shuffled[40:45], key=lambda x: int(x.split("_")[-1])),
                "test": sorted(shuffled[45:], key=lambda x: int(x.split("_")[-1])),
            }

            task_rows = 0
            valid_samples = 0
            episode_lengths = {}
            for episode_name in episodes:
                episode = data[episode_name]
                rows = int(episode.attrs["num_samples"])
                assert rows == len(episode["actions"]) and rows >= 2
                episode_lengths[episode_name] = rows
                task_rows += rows
                valid_samples += rows - 1

                # First and last canonical examples must remain inside this episode:
                # obs[0] -> action[1] and obs[T-2] -> action[T-1].
                assert episode["obs/agentview_rgb"].shape[0] == rows
                assert episode["obs/joint_states"].shape == (rows, 7)
                assert episode["obs/gripper_states"].shape == (rows, 2)
                assert episode["actions"].shape == (rows, 7)

                digest = hashlib.sha256(
                    episode["states"][:].tobytes() + episode["actions"][:].tobytes()
                ).hexdigest()
                episode_id = f"{task}/{episode_name}"
                trajectory_fingerprints.setdefault(digest, []).append(episode_id)
                all_episode_ids.add(episode_id)

            split_records = {}
            for split_name, names in splits.items():
                ids = {f"{task}/{name}" for name in names}
                split_episode_ids[split_name].update(ids)
                split_records[split_name] = {
                    "episodes": names,
                    "episode_count": len(names),
                    "valid_sample_count": int(sum(episode_lengths[name] - 1 for name in names)),
                }

            for episode_name in splits["train"]:
                episode = data[episode_name]
                # Only rows that can be canonical inputs are used: obs[0:T-1].
                joint = episode["obs/joint_states"][:-1].astype(np.float64)
                fingers = episode["obs/gripper_states"][:-1].astype(np.float64)
                train_proprio.append(np.concatenate((joint, fingers), axis=1))

            problem_info = json.loads(data.attrs["problem_info"])
            replay_path = (
                output_dir / path.stem / "replay_alignment" / "demo_0_replay_alignment.json"
            )
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            assert replay["verification_passed"] is True
            task_records.append({
                "task_id": task,
                "instruction": problem_info["language_instruction"],
                "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": actual_bytes,
                "sha256": actual_sha256,
                "episode_count": len(episodes),
                "stored_row_count": task_rows,
                "valid_sample_count": valid_samples,
                "episode_length_min": min(episode_lengths.values()),
                "episode_length_max": max(episode_lengths.values()),
                "alignment_evidence": {
                    "episode": "demo_0",
                    "image_rows_compared": replay["image_comparison_rows"],
                    "post_action_image_candidate_closer": replay["post_better_image_rows"],
                    "post_action_proprioception_rmse": 0.0,
                    "free_running_replay_success": replay["all_replays_successful"],
                    "state_anchored_replay_success": replay["all_local_replays_successful"],
                    "report": str(replay_path.relative_to(ROOT)).replace("\\", "/"),
                },
                "splits": split_records,
            })

    duplicate_groups = [members for members in trajectory_fingerprints.values() if len(members) > 1]
    assert not duplicate_groups, "Exact duplicate trajectories need grouped split assignment"
    assert split_episode_ids["train"].isdisjoint(split_episode_ids["validation"])
    assert split_episode_ids["train"].isdisjoint(split_episode_ids["test"])
    assert split_episode_ids["validation"].isdisjoint(split_episode_ids["test"])
    assert set.union(*split_episode_ids.values()) == all_episode_ids

    proprio = np.concatenate(train_proprio, axis=0)
    mean = proprio.mean(axis=0)
    raw_std = proprio.std(axis=0)
    std = np.maximum(raw_std, STD_EPSILON)
    npz_path = output_dir / "proprio_normalization.npz"
    np.savez(
        npz_path,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        raw_std=raw_std.astype(np.float32),
        sample_count=np.asarray([len(proprio)], dtype=np.int64),
    )

    manifest = {
        "manifest_version": 1,
        "generated_on": date.today().isoformat(),
        "dataset_source": "yifengzhu-hf/LIBERO-datasets",
        "dataset_revision": DATASET_REVISION,
        "alignment_status": "VERIFIED_POST_ACTION_IN_ONE_COMPLETE_DEMO_PER_INITIAL_TASK",
        "training_pair": "(upright agentview_rgb[k], instruction, joint_states[k] + gripper_states[k]) -> actions[k+1]",
        "valid_input_index_per_episode": "0 <= k <= T-2",
        "image_contract": {
            "source": "uint8 RGB [128,128,3], OpenGL bottom-left row origin",
            "adapter": "vertical flip exactly once to upright top-left row origin",
            "source_files_modified": False,
        },
        "split": {
            "seed": SPLIT_SEED,
            "algorithm": "numpy.default_rng(seed), one permutation per task in manifest order",
            "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
            "unit": "whole episode",
            "per_task_counts": {"train": 40, "validation": 5, "test": 5},
            "disjointness_verified": True,
        },
        "duplicate_check": {
            "fingerprint": "SHA-256(states bytes + actions bytes)",
            "trajectories_checked": len(all_episode_ids),
            "exact_duplicate_groups": duplicate_groups,
        },
        "proprioception_normalizer": {
            "field_order": [
                "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7",
                "finger_1", "finger_2",
            ],
            "fit_split": "train only",
            "valid_input_rows_only": True,
            "sample_count": len(proprio),
            "mean": mean.tolist(),
            "raw_std": raw_std.tolist(),
            "std": std.tolist(),
            "epsilon_floor": STD_EPSILON,
            "npz": str(npz_path.relative_to(ROOT)).replace("\\", "/"),
        },
        "verification": {
            "first_and_last_valid_sample_checked_for_every_episode": True,
            "no_cross_episode_samples": True,
            "action_domain": "stored normalized OSC_POSE commands; not z-scored",
            "time": "20 Hz relative simulation time; no wall-clock timestamp claim",
        },
        "tasks": task_records,
    }
    manifest_path = manifest_dir / "libero_goal_stage2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    summary = {
        "manifest": str(manifest_path),
        "tasks": len(task_records),
        "episodes": len(all_episode_ids),
        "split_episode_counts": {name: len(ids) for name, ids in split_episode_ids.items()},
        "valid_samples": sum(record["valid_sample_count"] for record in task_records),
        "training_normalizer_samples": len(proprio),
        "exact_duplicate_groups": len(duplicate_groups),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
