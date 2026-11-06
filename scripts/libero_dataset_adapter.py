"""Thin Stage 2 adapter around robomimic's SequenceDataset.

The adapter exposes the verified post-action LIBERO samples without loading a
tokenizer, CLIP, or any project neural network.

Windows CMD verification:
    venv\Scripts\python.exe scripts\libero_dataset_adapter.py
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import DataLoader

import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.dataset import SequenceDataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "libero_goal_stage2.json"
OBS_KEYS = ("agentview_rgb", "joint_states", "gripper_states")


class AlignedLiberoSequenceDataset:
    """Expose obs[k] -> action[k+1] from selected whole episodes."""

    def __init__(self, manifest_path: Path, task_id: str, split: str):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Unknown split: {split}")
        matches = [record for record in self.manifest["tasks"] if record["task_id"] == task_id]
        if len(matches) != 1:
            raise ValueError(f"Task {task_id!r} is not unique in {self.manifest_path}")
        self.task = matches[0]
        self.task_id = task_id
        self.split = split
        self.instruction = self.task["instruction"]
        self.hdf5_path = ROOT / self.task["file"]
        self.demos = self.task["splits"][split]["episodes"]

        normalizer = self.manifest["proprioception_normalizer"]
        self.proprio_mean = np.asarray(normalizer["mean"], dtype=np.float32)
        self.proprio_std = np.asarray(normalizer["std"], dtype=np.float32)
        assert self.proprio_mean.shape == self.proprio_std.shape == (9,)

        ObsUtils.initialize_obs_modality_mapping_from_dict({
            "rgb": ["agentview_rgb"],
            "low_dim": ["joint_states", "gripper_states"],
        })
        self.reader = SequenceDataset(
            hdf5_path=str(self.hdf5_path),
            obs_keys=OBS_KEYS,
            dataset_keys=("actions",),
            frame_stack=1,
            seq_length=2,
            pad_frame_stack=False,
            pad_seq_length=False,
            get_pad_mask=False,
            goal_mode=None,
            hdf5_cache_mode=None,
            hdf5_use_swmr=True,
            hdf5_normalize_obs=False,
            filter_by_attribute=None,
            load_next_obs=False,
        )
        # The public load_demo_info method rebuilds the index map using only
        # whole episodes from the manifest. No mask is written into the HDF5.
        self.reader.load_demo_info(demos=self.demos)
        self.reader.close_and_delete_hdf5_handle()
        assert len(self.reader) == self.task["splits"][split]["valid_sample_count"]

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, index: int) -> dict:
        sequence = self.reader[index]
        episode = self.reader._index_to_demo_id[index]
        row = index - self.reader._demo_id_to_start_indices[episode]

        # robomimic gives float32 RGB in [0,1] as [time, channel, height, width].
        # Source rows use OpenGL's bottom-left origin, so flip the height once.
        image = np.flip(sequence["obs"]["agentview_rgb"][0], axis=-2).copy()
        proprio_raw = np.concatenate((
            sequence["obs"]["joint_states"][0],
            sequence["obs"]["gripper_states"][0],
        )).astype(np.float32, copy=False)
        proprio = ((proprio_raw - self.proprio_mean) / self.proprio_std).astype(np.float32)
        action = sequence["actions"][1].astype(np.float32, copy=True)

        assert image.shape == (3, 128, 128)
        assert proprio.shape == (9,)
        assert action.shape == (7,)
        return {
            "image": image,
            "instruction": self.instruction,
            "proprio": proprio,
            "action": action,
            "metadata": {
                "task_id": self.task_id,
                "episode": episode,
                "observation_row": row,
                "action_row": row + 1,
            },
        }

    def index_for(self, episode: str, observation_row: int) -> int:
        if episode not in self.reader._demo_id_to_start_indices:
            raise KeyError(episode)
        length = self.reader._demo_id_to_demo_length[episode]
        if not 0 <= observation_row <= length - 2:
            raise IndexError((episode, observation_row, length))
        return self.reader._demo_id_to_start_indices[episode] + observation_row

    def close(self) -> None:
        self.reader.close_and_delete_hdf5_handle()


def expected_sample(file: h5py.File, episode: str, row: int, mean, std) -> tuple[np.ndarray, ...]:
    group = file[f"data/{episode}"]
    image = group["obs/agentview_rgb"][row]
    image = np.transpose(np.flip(image, axis=0).copy(), (2, 0, 1)).astype(np.float32) / 255.0
    proprio_raw = np.concatenate((
        group["obs/joint_states"][row].astype(np.float32),
        group["obs/gripper_states"][row].astype(np.float32),
    ))
    proprio = ((proprio_raw - mean) / std).astype(np.float32)
    action = group["actions"][row + 1].astype(np.float32)
    return image, proprio, action


def main() -> None:
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    checks = []
    total_boundary_samples = 0
    for task in manifest["tasks"]:
        for split in ("train", "validation", "test"):
            dataset = AlignedLiberoSequenceDataset(DEFAULT_MANIFEST, task["task_id"], split)
            try:
                with h5py.File(dataset.hdf5_path, "r", swmr=True, libver="latest") as file:
                    for episode in dataset.demos:
                        rows = int(file[f"data/{episode}"].attrs["num_samples"])
                        for row in sorted({0, rows - 2}):
                            sample = dataset[dataset.index_for(episode, row)]
                            expected = expected_sample(
                                file, episode, row, dataset.proprio_mean, dataset.proprio_std
                            )
                            np.testing.assert_array_equal(sample["image"], expected[0])
                            np.testing.assert_allclose(sample["proprio"], expected[1], rtol=0, atol=1e-6)
                            np.testing.assert_array_equal(sample["action"], expected[2])
                            assert sample["metadata"]["episode"] == episode
                            assert sample["metadata"]["observation_row"] == row
                            assert sample["metadata"]["action_row"] == row + 1
                            total_boundary_samples += 1

                loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=False, num_workers=0)
                batch = next(iter(loader))
                checks.append({
                    "task_id": task["task_id"],
                    "split": split,
                    "samples": len(dataset),
                    "batch_image_shape": list(batch["image"].shape),
                    "batch_instruction_count": len(batch["instruction"]),
                    "batch_proprio_shape": list(batch["proprio"].shape),
                    "batch_action_shape": list(batch["action"].shape),
                })
            finally:
                dataset.close()

    report = {
        "status": "PASSED",
        "adapter": "robomimic SequenceDataset(seq_length=2, pad_seq_length=False)",
        "pairing": "upright obs[k] + instruction + normalized proprio[k] -> action[k+1]",
        "image_output": "float32 RGB [3,128,128] in [0,1]; resize/checkpoint normalization deferred to Stage 3",
        "boundary_samples_checked": total_boundary_samples,
        "episodes_checked": sum(task["episode_count"] for task in manifest["tasks"]),
        "checks": checks,
    }
    output = ROOT / "outputs" / "stage2" / "adapter_verification.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "checks"}, indent=2))
    print("Report:", output)


if __name__ == "__main__":
    main()
