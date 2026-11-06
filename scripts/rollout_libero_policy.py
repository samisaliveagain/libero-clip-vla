"""Stage 5 development rollouts with official LIBERO reset states and success.

Windows CMD: venv\Scripts\python.exe scripts\rollout_libero_policy.py
"""

from __future__ import annotations

import argparse
import json
import hashlib
import random
import os
import re
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from clip_features import FrozenCLIPFeatures, upright_live_image
from libero_policy import LiberoActionPolicy


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "LIBERO"))
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))

from libero.libero import get_libero_path  # noqa: E402
from libero.libero.benchmark import get_benchmark  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from libero.libero.envs import bddl_utils  # noqa: E402


TASK = "put_the_bowl_on_the_plate"
OUTPUT = ROOT / "outputs" / "stage5" / TASK / "development_rollouts"
CHECKPOINTS = ROOT / "outputs" / "stage5" / TASK / "baseline" / "checkpoints"


def selected_checkpoint() -> Path:
    candidates = []
    for path in CHECKPOINTS.glob("epoch*-val*.ckpt"):
        match = re.search(r"-val([0-9.]+)\.ckpt$", path.name)
        if match:
            candidates.append((float(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No validated epoch checkpoint in {CHECKPOINTS}")
    return min(candidates)[1]


def load_policy(path: Path, device: torch.device) -> LiberoActionPolicy:
    # These checkpoints are produced by this project's local Lightning run.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    weights = {name.removeprefix("policy."): value for name, value in
               checkpoint["state_dict"].items() if name.startswith("policy.")}
    policy = LiberoActionPolicy().to(device)
    policy.load_state_dict(weights, strict=True)
    policy.eval()
    return policy


def policy_input(observation: dict, mean: np.ndarray, std: np.ndarray):
    rgb = upright_live_image(observation["agentview_image"])
    image = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32)[None] / 255.0)
    raw = np.concatenate((
        observation["robot0_joint_pos"], observation["robot0_gripper_qpos"]
    )).astype(np.float32)
    proprio = torch.from_numpy(((raw - mean) / std)[None])
    return rgb, image, proprio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--reset-indices", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/libero_goal_stage2.json")
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--scene-task", default=None,
                        help="Use this task's scene and reset list, scoring the requested task goal")
    parser.add_argument("--instruction-mode", choices=["correct", "blank", "shuffled"], default="correct")
    parser.add_argument("--instruction-task", default=None)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--settling-steps", type=int, default=5)
    parser.add_argument("--settling-gripper", type=int, choices=[-1, 0], default=0,
                        help="0: pinned LIBERO protocol; -1: explicit open-gripper startup variant")
    args = parser.parse_args()
    checkpoint_path = args.checkpoint or selected_checkpoint()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise FileExistsError(f"Preserve existing evidence; choose a new --output: {output}")
    manifest = json.loads(args.manifest.read_text())
    norm = manifest["proprioception_normalizer"]
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    task_info = next(item for item in manifest["tasks"] if item["task_id"] == args.task)
    instruction = task_info["instruction"]
    if args.instruction_mode == "blank":
        instruction = ""
    elif args.instruction_mode == "shuffled":
        if not args.instruction_task or args.instruction_task == args.task:
            raise ValueError("Shuffled mode requires a different --instruction-task")
        instruction = next(item["instruction"] for item in manifest["tasks"]
                           if item["task_id"] == args.instruction_task)

    benchmark = get_benchmark("libero_goal")()
    task_index = benchmark.get_task_names().index(args.scene_task or args.task)
    goal_index = benchmark.get_task_names().index(args.task)
    goal_bddl = benchmark.get_task_bddl_file_path(goal_index)
    goal_states = bddl_utils.robosuite_parse_problem(str(goal_bddl))["goal_state"]
    goal_catalog = {
        item["task_id"]: bddl_utils.robosuite_parse_problem(str(benchmark.get_task_bddl_file_path(
            benchmark.get_task_names().index(item["task_id"]))))["goal_state"]
        for item in manifest["tasks"]
    }
    # The pinned LIBERO .pruned_init file contains NumPy objects. Its upstream
    # loader assumes torch.load's pre-2.6 default, so state that setting here
    # only for this trusted, project-local LIBERO artifact.
    init_path = (Path(get_libero_path("init_states")) / "libero_goal"
                 / benchmark.get_task(task_index).init_states_file)
    initial_states = torch.load(init_path, weights_only=False)
    if max(args.reset_indices) >= len(initial_states) or min(args.reset_indices) < 0:
        raise ValueError("Reset index outside the official task initial-state list")
    bddl = benchmark.get_task_bddl_file_path(task_index)
    features = FrozenCLIPFeatures()
    policy = load_policy(checkpoint_path, features.device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    # The instruction is constant through a rollout; its frozen features are cached.
    with torch.inference_mode():
        features.encode_text([instruction])
    results = []
    for reset_index in args.reset_indices:
        reset_seed = args.seed + reset_index
        random.seed(reset_seed)
        np.random.seed(reset_seed)
        torch.manual_seed(reset_seed)
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), robots=["Panda"], controller="OSC_POSE",
            camera_names=["agentview"], camera_heights=128,
            camera_widths=128, control_freq=20,
        )
        video = cv2.VideoWriter(
            str(output / f"reset_{reset_index:03d}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (128, 128),
        )
        if not video.isOpened():
            raise RuntimeError("Could not open rollout video writer")
        latencies = []
        success = False
        stop_reason = "horizon"
        step_count = 0
        actions, simulation_ms = [], []
        goals_reached = set()
        initial_image_sha256 = initial_state_sha256 = None
        started = time.perf_counter()
        try:
            env.seed(reset_seed)
            env.reset()
            observation = env.set_init_state(initial_states[reset_index])
            settling_action = np.zeros(7, dtype=np.float32)
            settling_action[6] = args.settling_gripper
            for _ in range(args.settling_steps):
                observation, _, _, _ = env.step(settling_action)
            initial_image_sha256 = hashlib.sha256(observation["agentview_image"].tobytes()).hexdigest()
            initial_state_sha256 = hashlib.sha256(env.get_sim_state().tobytes()).hexdigest()
            for goal in goal_states:
                for name in goal[1:]:
                    if name not in env.env.object_states_dict:
                        raise ValueError(f"Requested goal object {name} is absent from scene")
            for step in range(args.horizon):
                rgb, image, proprio = policy_input(observation, mean, std)
                if step % 2 == 0:
                    video.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                start = time.perf_counter()
                with torch.inference_mode():
                    visual = features.encode_images(image)
                    text, mask = features.encode_text([instruction])
                    action = policy(visual, text, mask, proprio.to(features.device))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)
                command = action[0].float().cpu().numpy()
                if command.shape != (7,) or not np.isfinite(command).all() or np.any(np.abs(command) > 1):
                    stop_reason = "invalid_action"
                    break
                actions.append(command.copy())
                sim_start = time.perf_counter()
                observation, _, _, _ = env.step(command.astype(np.float32))
                simulation_ms.append((time.perf_counter() - sim_start) * 1000)
                step_count = step + 1
                # Privileged simulator predicates are evaluation metadata only.
                # They never enter policy_input, CLIP, or the learned policy.
                for goal_name, predicates in goal_catalog.items():
                    available = all(name in env.env.object_states_dict
                                    for predicate in predicates for name in predicate[1:])
                    if available and all(env.env._eval_predicate(predicate) for predicate in predicates):
                        goals_reached.add(goal_name)
                if all(env.env._eval_predicate(goal) for goal in goal_states):
                    success = True
                    stop_reason = "success"
                    break
        except Exception as exc:
            stop_reason = f"fault: {type(exc).__name__}: {exc}"
        finally:
            if step_count:
                final_rgb = upright_live_image(observation["agentview_image"])
                video.write(cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(output / f"reset_{reset_index:03d}_final.png"),
                            cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR))
            video.release()
            env.close()
        result = {
            "reset_index": reset_index, "success": success,
            "seed": reset_seed,
            "initial_image_sha256": initial_image_sha256,
            "initial_state_sha256": initial_state_sha256,
            "stop_reason": stop_reason, "steps": step_count,
            "goals_reached": sorted(goals_reached),
            "inference_ms_median": statistics.median(latencies) if latencies else None,
            "inference_ms_p95": float(np.percentile(latencies, 95)) if latencies else None,
            "simulation_and_render_ms_median": statistics.median(simulation_ms) if simulation_ms else None,
            "simulation_and_render_ms_p95": float(np.percentile(simulation_ms, 95)) if simulation_ms else None,
            "wall_seconds": time.perf_counter() - started,
        }
        np.savez_compressed(output / f"reset_{reset_index:03d}_trace.npz",
                            actions=np.asarray(actions), inference_ms=np.asarray(latencies),
                            simulation_ms=np.asarray(simulation_ms))
        results.append(result)
        print(json.dumps(result), flush=True)
        (output / "progress.json").write_text(json.dumps(results, indent=2)+"\n")
    report = {
        "checkpoint": str(checkpoint_path), "task": args.task,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "manifest": str(args.manifest),
        "scene_task": args.scene_task or args.task, "goal_predicates": goal_states,
        "instruction_mode": args.instruction_mode, "seed_base": args.seed,
        "instruction": instruction, "reset_indices": args.reset_indices,
        "horizon": args.horizon, "settling_steps": args.settling_steps,
        "settling_gripper": args.settling_gripper,
        "success_count": sum(item["success"] for item in results),
        "rollout_count": len(results), "results": results,
        "peak_gpu_allocated_mib": (
            torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
