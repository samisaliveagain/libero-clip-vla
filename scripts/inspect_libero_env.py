"""Inspect the Stage 1 LIBERO Panda environment without loading a dataset."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBERO_CHECKOUT = PROJECT_ROOT / "third_party" / "LIBERO"
LIBERO_CONFIG = PROJECT_ROOT / ".libero"

# The pinned LIBERO revision uses a nested source layout that its editable
# installation does not expose correctly on Windows.
sys.path.insert(0, str(LIBERO_CHECKOUT))
os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_CONFIG))

from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


def main() -> None:
    task_file = (
        Path(get_libero_path("bddl_files"))
        / "libero_goal"
        / "put_the_bowl_on_the_plate.bddl"
    )
    output_file = PROJECT_ROOT / "stage1_libero_inspection.png"

    env = OffScreenRenderEnv(
        bddl_file_name=str(task_file),
        robots=["Panda"],
        controller="OSC_POSE",
        camera_names=["agentview"],
        camera_heights=128,
        camera_widths=128,
        control_freq=20,
    )

    try:
        observation = env.reset()
        action_low, action_high = env.env.action_spec

        print("Language:", env.language_instruction)
        print("Robot:", env.robots[0].robot_model.name)
        print("Controller:", env.robots[0].controller.name)
        print("Action shape:", action_low.shape)
        print("Action bounds:", action_low, action_high)

        selected_keys = (
            "agentview_image",
            "robot0_joint_pos",
            "robot0_gripper_qpos",
            "robot0_eef_pos",
            "robot0_eef_quat",
        )
        print(
            "Selected observation shapes:",
            {key: observation[key].shape for key in selected_keys},
        )

        image = observation["agentview_image"]
        Image.fromarray(image).save(output_file)
        print("Image dtype and range:", image.dtype, (int(image.min()), int(image.max())))
        print("Saved image:", output_file)
        print("Initial task success:", env.check_success())

        position_before = observation["robot0_eef_pos"].copy()
        next_observation, reward, done, _ = env.step(np.zeros(7, dtype=np.float32))
        zero_action_drift = np.linalg.norm(
            next_observation["robot0_eef_pos"] - position_before
        )
        print("Zero-action reward:", reward)
        print("Zero-action done:", done)
        print("Zero-action EEF drift (m):", zero_action_drift)
        print("Simulation time after one step (s):", env.sim.data.time)
    finally:
        env.close()


if __name__ == "__main__":
    main()
