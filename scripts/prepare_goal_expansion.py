"""Audit seven added Goal datasets and preserve the original episode splits.

NON-LEARNED INFRASTRUCTURE. Original HDF5 files remain read-only.
"""
import hashlib
import json
import argparse
import time
from pathlib import Path

import h5py
import numpy as np

from prepare_libero_stage2_data import ROOT, ordered_episodes, sha256_file
from verify_libero_replay_alignment import create_environment, verify_episode
from libero_dataset_adapter import AlignedLiberoSequenceDataset, expected_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-download", action="store_true")
    parser.add_argument("--verify-boundaries-only", action="store_true")
    args = parser.parse_args()
    if args.verify_boundaries_only:
        verify_boundaries(ROOT / "data/manifests/libero_goal_stage6.json", ROOT / "outputs/stage6/data_audit")
        return
    provenance_path = ROOT / "outputs/stage6/download_provenance.json"
    deadline = time.monotonic() + 3 * 3600
    while args.wait_for_download and not provenance_path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Dataset download did not complete within three hours")
        time.sleep(30)
    output = ROOT / "outputs/stage6/data_audit"
    output.mkdir(parents=True, exist_ok=True)
    original_path = ROOT / "data/manifests/libero_goal_stage2.json"
    manifest = json.loads(original_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    records = {record["task_id"]: record for record in manifest["tasks"]}
    fingerprints = {}
    for source in provenance["files"]:
        path = ROOT / "data/datasets" / source["file"]
        task = path.stem.removesuffix("_demo")
        assert sha256_file(path) == source["sha256"]
        with h5py.File(path, "r") as file:
            data = file["data"]
            episodes = ordered_episodes(data)
            assert len(episodes) == 50
            lengths = {}
            for name in episodes:
                demo = data[name]
                n = len(demo["actions"])
                lengths[name] = n
                assert demo["obs/agentview_rgb"].shape == (n, 128, 128, 3)
                assert demo["obs/agentview_rgb"].dtype == np.uint8
                for key, width in [("actions", 7), ("obs/joint_states", 7), ("obs/gripper_states", 2)]:
                    value = demo[key][:]
                    assert value.shape == (n, width) and np.isfinite(value).all()
                assert np.abs(demo["actions"][:]).max() <= 1 + 1e-6
                digest = hashlib.sha256(demo["states"][:].tobytes()+demo["actions"][:].tobytes()).hexdigest()
                fingerprints.setdefault(digest, []).append(f"{task}/{name}")
            if task in records:
                continue
            print("AUDIT", task, flush=True)
            env = create_environment(data)
            try:
                evidence = verify_episode(env, data["demo_0"], output / task, save_visuals=True)
                assert evidence["post_better_image_rows"] == evidence["image_comparison_rows"]
                assert evidence["local_replay_success"]
                joint_idx = np.asarray(env.robots[0]._ref_joint_pos_indexes) + 1
                gripper_idx = np.asarray(env.robots[0]._ref_gripper_joint_pos_indexes) + 1
                for name in episodes:
                    demo = data[name]
                    states = demo["states"][:]
                    np.testing.assert_array_equal(states[1:, joint_idx], demo["obs/joint_states"][:-1])
                    np.testing.assert_array_equal(states[1:, gripper_idx], demo["obs/gripper_states"][:-1])
            finally:
                env.close()
            (output / f"{task}.json").write_text(json.dumps(evidence, indent=2)+"\n")
            # Task-keyed seeds keep membership stable if task order changes.
            task_seed = int(hashlib.sha256(task.encode()).hexdigest()[:8], 16) ^ 20260920
            shuffled = np.random.default_rng(task_seed).permutation(episodes).tolist()
            splits = {}
            for split, names in [("train", shuffled[:40]), ("validation", shuffled[40:45]), ("test", shuffled[45:])]:
                splits[split] = dict(episodes=names, episode_count=len(names),
                                     valid_sample_count=sum(lengths[n]-1 for n in names))
            records[task] = dict(task_id=task, instruction=json.loads(data.attrs["problem_info"])["language_instruction"],
                                 file=str(path.relative_to(ROOT)).replace("\\", "/"), bytes=path.stat().st_size,
                                 sha256=source["sha256"], episode_count=50, stored_row_count=sum(lengths.values()),
                                 valid_sample_count=sum(lengths.values())-50, splits=splits,
                                 split_seed=task_seed, alignment_evidence=str(output / f"{task}.json"))
    duplicates = [v for v in fingerprints.values() if len(v) > 1]
    assert not duplicates, duplicates
    manifest["manifest_version"] = 2
    manifest["tasks"] = list(records.values())
    manifest["parent_manifest_sha256"] = sha256_file(original_path)
    manifest["alignment_status"] = "ALL_TASKS_REPLAY_AUDITED_ALL_EPISODE_PROPRIO_ALIGNMENT_CHECKED_FOR_ADDED_TASKS"
    manifest["split"]["algorithm"] += "; added tasks: SHA256(task) first 32 bits XOR 20260920"
    manifest["duplicate_check"] = dict(trajectories_checked=len(fingerprints), exact_duplicate_groups=duplicates)
    manifest["proprioception_normalizer"]["reuse_note"] = (
        "Original three-task training-only normalizer retained for warm-start compatibility; no validation/test fitting.")
    manifest_path = ROOT / "data/manifests/libero_goal_stage6.json"
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    verify_boundaries(manifest_path, output)


def verify_boundaries(manifest_path, output):
    manifest = json.loads(manifest_path.read_text())
    boundaries = 0
    controller_reference = None
    controller_checks = []
    for task in manifest["tasks"]:
        with h5py.File(ROOT / task["file"], "r") as source:
            kwargs = json.loads(source["data"].attrs["env_args"])["env_kwargs"]
            assert kwargs["control_freq"] == 20
            assert kwargs["robots"] == ["Panda"]
            assert kwargs["camera_heights"] == kwargs["camera_widths"] == 128
            controller = kwargs["controller_configs"]
            assert controller["type"] == "OSC_POSE" and controller["control_delta"]
            if controller_reference is None:
                controller_reference = controller
            assert controller == controller_reference, task["task_id"]
            controller_checks.append(task["task_id"])
        sets = [set(task["splits"][s]["episodes"]) for s in ["train", "validation", "test"]]
        assert len(set.union(*sets)) == sum(map(len, sets)) == 50
        for split in ["train", "validation", "test"]:
            dataset = AlignedLiberoSequenceDataset(manifest_path, task["task_id"], split)
            try:
                with h5py.File(dataset.hdf5_path, "r", swmr=True, libver="latest") as file:
                    for episode in dataset.demos:
                        for row in [0, len(file[f"data/{episode}/actions"])-2]:
                            got = dataset[dataset.index_for(episode, row)]
                            expected = expected_sample(file, episode, row, dataset.proprio_mean, dataset.proprio_std)
                            for key, value in zip(["image", "proprio", "action"], expected):
                                np.testing.assert_allclose(got[key], value, rtol=0, atol=1e-6)
                            boundaries += 1
            finally:
                dataset.close()
    report = dict(status="PASSED", tasks=len(manifest["tasks"]), episodes=manifest["duplicate_check"]["trajectories_checked"],
                  boundary_samples=boundaries, manifest=str(manifest_path),
                  controller_identical_across_tasks=controller_checks,
                  controller=controller_reference, control_freq=20,
                  duplicate_groups=manifest["duplicate_check"]["exact_duplicate_groups"])
    (output / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
