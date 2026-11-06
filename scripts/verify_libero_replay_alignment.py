"""Stage 2: verify LIBERO replay and observation/action alignment.

NON-LEARNED INFRASTRUCTURE. This opens the original HDF5 read-only, restores
each recorded MuJoCo model/state, and compares stored observations with both
pre-action and post-action simulator states.

Windows CMD:
    venv\Scripts\python.exe scripts\verify_libero_replay_alignment.py --demo demo_0
    venv\Scripts\python.exe scripts\verify_libero_replay_alignment.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import h5py
import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIBERO_CHECKOUT = ROOT / "third_party" / "LIBERO"
LIBERO_CONFIG = ROOT / ".libero"
os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_CONFIG))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(LIBERO_CHECKOUT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import TASK_MAPPING  # noqa: E402
from libero.libero.utils.utils import postprocess_model_xml  # noqa: E402


DEFAULT_DATASET = ROOT / "data/datasets/libero_goal/put_the_bowl_on_the_plate_demo.hdf5"


def restore_state(env, state: np.ndarray):
    """Restore a recorded state and force all sensors to observe that state."""
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    return env._get_observations()


def errors(actual: np.ndarray, expected: np.ndarray) -> dict[str, float | int]:
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "exact": bool(np.array_equal(actual, expected)),
    }


def upright_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a LIBERO OpenGL frame to normal top-left display."""
    return np.flip(image, axis=0).copy()


def create_environment(data_group):
    env_args = json.loads(data_group.attrs["env_args"])
    kwargs = dict(env_args["env_kwargs"])
    kwargs.update(
        bddl_file_name=str(
            Path(get_libero_path("bddl_files"))
            / "libero_goal"
            / Path(data_group.attrs["bddl_file_name"]).name
        ),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview"],
        camera_heights=128,
        camera_widths=128,
        camera_depths=False,
        camera_segmentations=None,
        control_freq=20,
        ignore_done=True,
    )
    return TASK_MAPPING[env_args["problem_name"]](**kwargs)


def reset_episode(env, episode, restore_gripper_target: bool = False):
    model_xml = postprocess_model_xml(episode.attrs["model_file"], {})
    # Published demonstrations retain absolute paths from the data author's
    # checkout for LIBERO objects. Repair only that known prefix; robosuite's
    # helper above has already repaired its own asset paths.
    tree = ET.fromstring(model_xml)
    libero_assets = LIBERO_CHECKOUT / "libero" / "libero" / "assets"
    for element in tree.find("asset"):
        old_path = element.get("file")
        marker = "/chiliocosm/assets/"
        if old_path and marker in old_path.replace("\\", "/"):
            relative = old_path.replace("\\", "/").split(marker, 1)[1]
            element.set("file", (libero_assets / relative).resolve().as_posix())
    model_xml = ET.tostring(tree, encoding="utf-8").decode("utf-8")
    env.reset_from_xml_string(model_xml)
    env.sim.reset()
    obs = restore_state(env, episode["states"][0])
    if restore_gripper_target:
        # create_dataset.py discards five already-executed steps, but the HDF5
        # does not store PandaGripper.current_action. At the first retained row
        # both fingers are at their open limits, so reconstruct the accumulated
        # actuator target as [open-left, open-right] = [+1, -1].
        first_fingers = episode["obs/gripper_states"][0]
        env.robots[0].gripper.current_action = np.sign(first_fingers).astype(np.float64)
    # Match upstream create_dataset.py: controller/gripper internals were reset
    # before the recorded simulator state was installed.
    return obs


def verify_episode(env, episode, output: Path, save_visuals: bool) -> dict:
    name = episode.name.rsplit("/", 1)[-1]
    states = episode["states"][:]
    actions = episode["actions"][:]
    stored_images = episode["obs/agentview_rgb"][:]
    stored_joints = episode["obs/joint_states"][:]
    stored_gripper = episode["obs/gripper_states"][:]
    n = len(actions)

    # Independent state restoration: stored observation row k should look like
    # state k+1 if it was captured after action k, and like state k if pre-action.
    pre_image_mae, post_image_mae = [], []
    pre_images, post_images = [], []
    reset_episode(env, episode)
    joint_qpos = np.asarray(env.robots[0]._ref_joint_pos_indexes)
    gripper_qpos = np.asarray(env.robots[0]._ref_gripper_joint_pos_indexes)
    recorded_joint_from_state = states[:, 1 + joint_qpos]
    recorded_gripper_from_state = states[:, 1 + gripper_qpos]
    pre_joint_rmse = np.sqrt(np.mean((recorded_joint_from_state[:-1] - stored_joints[:-1]) ** 2, axis=1))
    post_joint_rmse = np.sqrt(np.mean((recorded_joint_from_state[1:] - stored_joints[:-1]) ** 2, axis=1))
    pre_gripper_rmse = np.sqrt(np.mean((recorded_gripper_from_state[:-1] - stored_gripper[:-1]) ** 2, axis=1))
    post_gripper_rmse = np.sqrt(np.mean((recorded_gripper_from_state[1:] - stored_gripper[:-1]) ** 2, axis=1))

    image_indices = list(range(n - 1)) if save_visuals else sorted({0, (n - 2) // 2, n - 2})
    for k in image_indices:
        pre = restore_state(env, states[k])
        pre_images.append(pre["agentview_image"].copy())
        pre_image_mae.append(errors(pre["agentview_image"], stored_images[k])["mae"])

        post = restore_state(env, states[k + 1])
        post_images.append(post["agentview_image"].copy())
        post_image_mae.append(errors(post["agentview_image"], stored_images[k])["mae"])

    # Sequential replay tests the actual controller chain, not state playback.
    reset_episode(env, episode, restore_gripper_target=True)
    replay_state_l2, replay_image_mae, replay_joint_rmse, replay_gripper_rmse = [], [], [], []
    replay_frames = []
    success_steps = []
    final_reward = None
    for k, action in enumerate(actions):
        observation, reward, _, _ = env.step(action)
        replay_frames.append(observation["agentview_image"].copy())
        replay_image_mae.append(errors(observation["agentview_image"], stored_images[k])["mae"])
        replay_joint_rmse.append(errors(observation["robot0_joint_pos"], stored_joints[k])["rmse"])
        replay_gripper_rmse.append(errors(observation["robot0_gripper_qpos"], stored_gripper[k])["rmse"])
        if k < n - 1:
            replay_state_l2.append(float(np.linalg.norm(env.sim.get_state().flatten() - states[k + 1])))
        if env._check_success():
            success_steps.append(k)
        final_reward = float(reward)
    free_final_success = bool(env._check_success())

    # Teacher-forced action replay: restore the exact recorded pre-action
    # physical state at every row while preserving the reconstructed gripper
    # target. This tests each controller transition without allowing tiny
    # contact differences to compound through the rest of the trajectory.
    reset_episode(env, episode, restore_gripper_target=True)
    local_state_l2, local_image_mae = [], []
    local_frames = []
    local_success_steps = []
    for k, action in enumerate(actions):
        if k:
            restore_state(env, states[k])
        observation, _, _, _ = env.step(action)
        local_frames.append(observation["agentview_image"].copy())
        local_image_mae.append(errors(observation["agentview_image"], stored_images[k])["mae"])
        if k < n - 1:
            local_state_l2.append(float(np.linalg.norm(env.sim.get_state().flatten() - states[k + 1])))
        if env._check_success():
            local_success_steps.append(k)

    result = {
        "episode": name,
        "rows": n,
        "proprioception_comparison_rows": n - 1,
        "image_comparison_rows": len(image_indices),
        "pre_state_image_mae_mean": float(np.mean(pre_image_mae)),
        "post_state_image_mae_mean": float(np.mean(post_image_mae)),
        "post_better_image_rows": int(np.sum(np.asarray(post_image_mae) < np.asarray(pre_image_mae))),
        "pre_state_joint_rmse_mean": float(np.mean(pre_joint_rmse)),
        "post_state_joint_rmse_mean": float(np.mean(post_joint_rmse)),
        "pre_state_gripper_rmse_mean": float(np.mean(pre_gripper_rmse)),
        "post_state_gripper_rmse_mean": float(np.mean(post_gripper_rmse)),
        "replay_state_l2_max": float(np.max(replay_state_l2)),
        "replay_state_l2_mean": float(np.mean(replay_state_l2)),
        "replay_image_mae_mean": float(np.mean(replay_image_mae)),
        "replay_image_mae_max": float(np.max(replay_image_mae)),
        "replay_joint_rmse_mean": float(np.mean(replay_joint_rmse)),
        "replay_gripper_rmse_mean": float(np.mean(replay_gripper_rmse)),
        "success": bool(success_steps),
        "first_success_row": success_steps[0] if success_steps else None,
        "final_success": free_final_success,
        "final_reward": final_reward,
        "local_replay_state_l2_max": float(np.max(local_state_l2)),
        "local_replay_state_l2_mean": float(np.mean(local_state_l2)),
        "local_replay_image_mae_mean": float(np.mean(local_image_mae)),
        "local_replay_success": bool(local_success_steps),
        "local_replay_first_success_row": local_success_steps[0] if local_success_steps else None,
        "local_replay_final_success": bool(env._check_success()),
    }

    if save_visuals:
        output.mkdir(parents=True, exist_ok=True)
        indices = sorted(set([0, min(32, n - 2), min(50, n - 2), n - 2]))
        canvas = np.full((len(indices) * 128, 3 * 128, 3), 255, dtype=np.uint8)
        for row, k in enumerate(indices):
            canvas[row * 128:(row + 1) * 128, 0:128] = upright_rgb(pre_images[k])
            canvas[row * 128:(row + 1) * 128, 128:256] = upright_rgb(stored_images[k])
            canvas[row * 128:(row + 1) * 128, 256:384] = upright_rgb(post_images[k])
        cv2.imwrite(str(output / f"{name}_alignment_pre_stored_post_upright.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
        x = np.arange(n - 1)
        axes[0].plot(x, pre_image_mae, label="stored vs state[k] (pre-action candidate)")
        axes[0].plot(x, post_image_mae, label="stored vs state[k+1] (post-action candidate)")
        axes[0].set_ylabel("RGB mean absolute error\n(pixel levels 0-255)")
        axes[0].legend()
        axes[0].grid(alpha=0.2)
        axes[1].plot(np.arange(len(replay_state_l2)), replay_state_l2, label="replayed post-step vs state[k+1]")
        axes[1].set_ylabel("Simulator state L2 error")
        axes[1].set_xlabel("Stored row k")
        axes[1].legend()
        axes[1].grid(alpha=0.2)
        fig.suptitle(f"{name}: alignment evidence and controller replay")
        fig.savefig(output / f"{name}_alignment_metrics.png", dpi=145)
        plt.close(fig)

        video = output / f"{name}_free_running_action_replay_upright.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 20, (128, 128))
        if not writer.isOpened():
            raise RuntimeError("Could not open replay MP4 writer")
        try:
            for frame in replay_frames:
                writer.write(cv2.cvtColor(upright_rgb(frame), cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

        local_video = output / f"{name}_state_anchored_action_replay_upright.mp4"
        writer = cv2.VideoWriter(str(local_video), cv2.VideoWriter_fourcc(*"mp4v"), 20, (128, 128))
        if not writer.isOpened():
            raise RuntimeError("Could not open state-anchored replay MP4 writer")
        try:
            for frame in local_frames:
                writer.write(cv2.cvtColor(upright_rgb(frame), cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", help="Verify one episode, e.g. demo_0")
    group.add_argument("--all", action="store_true", help="Verify all episodes")
    args = parser.parse_args()
    output = args.output or ROOT / "outputs" / "stage2" / args.file.stem / "replay_alignment"

    with h5py.File(args.file, "r") as file:
        data = file["data"]
        episode_names = sorted(data.keys(), key=lambda s: int(s.split("_")[-1])) if args.all else [args.demo]
        env = create_environment(data)
        try:
            results = []
            for index, name in enumerate(episode_names):
                print(f"[{index + 1}/{len(episode_names)}] {name}", flush=True)
                results.append(verify_episode(env, data[name], output, save_visuals=(name == "demo_0")))
        finally:
            env.close()

    summary = {
        "dataset": str(args.file.resolve()),
        "source_image_convention": "opengl (bottom-left row origin)",
        "inspection_image_convention": "upright (top-left row origin)",
        "alignment_rule": "stored obs[k] is post-action-k; canonical input obs[k] targets action[k+1]",
        "canonical_valid_rows_per_episode": "k=0..T-2; drop action[0] and final obs[T-1]",
        "episode_count": len(results),
        "all_replays_successful": all(r["success"] for r in results),
        "all_final_states_successful": all(r["final_success"] for r in results),
        "all_local_replays_successful": all(r["local_replay_success"] for r in results),
        "all_local_final_states_successful": all(r["local_replay_final_success"] for r in results),
        "post_better_image_rows": sum(r["post_better_image_rows"] for r in results),
        "image_comparison_rows": sum(r["image_comparison_rows"] for r in results),
        "proprioception_comparison_rows": sum(r["proprioception_comparison_rows"] for r in results),
        "max_replay_state_l2": max(r["replay_state_l2_max"] for r in results),
        "mean_pre_image_mae": float(np.mean([r["pre_state_image_mae_mean"] for r in results])),
        "mean_post_image_mae": float(np.mean([r["post_state_image_mae_mean"] for r in results])),
        "mean_replay_image_mae": float(np.mean([r["replay_image_mae_mean"] for r in results])),
        "max_local_replay_state_l2": max(r["local_replay_state_l2_max"] for r in results),
        "episodes": results,
    }
    if summary["post_better_image_rows"] != summary["image_comparison_rows"]:
        raise AssertionError("Post-action image alignment was not better for every sampled row")
    if any(r["post_state_joint_rmse_mean"] != 0.0 or r["post_state_gripper_rmse_mean"] != 0.0 for r in results):
        raise AssertionError("Stored proprioception did not exactly match state[k+1]")
    if not summary["all_local_replays_successful"] or not summary["all_local_final_states_successful"]:
        raise AssertionError("At least one state-anchored action replay failed")
    summary["verification_passed"] = True
    output.mkdir(parents=True, exist_ok=True)
    report_name = "all_episodes_replay_alignment.json" if args.all else f"{args.demo}_replay_alignment.json"
    (output / report_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))
    print("Report:", output / report_name)


if __name__ == "__main__":
    main()
