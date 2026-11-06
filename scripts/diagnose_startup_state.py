"""Compare retained demonstration startup with two explicit settling protocols."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

from rollout_libero_policy import ROOT, get_benchmark, get_libero_path, OffScreenRenderEnv
from clip_features import upright_live_image


def main():
    manifest = json.loads((ROOT / "data/manifests/libero_goal_stage6.json").read_text())
    starts = {}
    for task in manifest["tasks"]:
        with h5py.File(ROOT / task["file"], "r") as file:
            fingers = np.stack([file[f"data/{name}/obs/gripper_states"][0]
                                for name in task["splits"]["train"]["episodes"]])
            joints = np.stack([file[f"data/{name}/obs/joint_states"][0]
                               for name in task["splits"]["train"]["episodes"]])
            model = ET.fromstring(file["data/demo_0"].attrs["model_file"])
            camera = model.find(".//camera[@name='agentview']")
            starts[task["task_id"]] = dict(mean=fingers.mean(0).tolist(),
                                           minimum=fingers.min(0).tolist(), maximum=fingers.max(0).tolist(),
                                           joint_mean=joints.mean(0).tolist(),
                                           joint_minimum=joints.min(0).tolist(), joint_maximum=joints.max(0).tolist(),
                                           recorded_camera=dict(camera.attrib))
    suite = get_benchmark("libero_goal")()
    index = suite.get_task_names().index("put_the_bowl_on_the_plate")
    states = torch.load(Path(get_libero_path("init_states")) / "libero_goal" /
                        suite.get_task(index).init_states_file, weights_only=False)
    output = ROOT / "outputs/stage6/startup_diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    measurements = []
    for settling_steps, gripper_command in [(5, 0), (5, -1), (10, -1)]:
        np.random.seed(20261007)
        env = OffScreenRenderEnv(bddl_file_name=str(suite.get_task_bddl_file_path(index)),
                                 robots=["Panda"], controller="OSC_POSE", camera_names=["agentview"],
                                 camera_heights=128, camera_widths=128, control_freq=20)
        try:
            env.seed(20261007)
            env.reset()
            obs = env.set_init_state(states[0])
            before = obs["robot0_gripper_qpos"].tolist()
            command = np.zeros(7, dtype=np.float32)
            command[-1] = gripper_command
            for _ in range(settling_steps):
                obs, _, _, _ = env.step(command)
            measurements.append(dict(settling_steps=settling_steps, settling_action=command.tolist(), initial_fingers=before,
                                     settled_fingers=obs["robot0_gripper_qpos"].tolist(),
                                     settled_joints=obs["robot0_joint_pos"].tolist(),
                                     camera_position=env.sim.model.cam_pos[env.sim.model.camera_name2id("agentview")].tolist(),
                                     camera_quaternion=env.sim.model.cam_quat[env.sim.model.camera_name2id("agentview")].tolist()))
            cv2.imwrite(str(output / f"settling_{settling_steps}_gripper_{gripper_command}.png"),
                        cv2.cvtColor(upright_live_image(obs["agentview_image"]), cv2.COLOR_RGB2BGR))
        finally:
            env.close()
    report = dict(training_episode_first_retained_fingers=starts, reset_zero=measurements,
                  zero_protocol_source="Pinned LIBERO lifelong/metric.py, five all-zero actions",
                  open_gripper_reference="https://github.com/openvla/openvla/blob/main/experiments/robot/libero/libero_utils.py",
                  note="Diagnostic only; the -1 variant is not the original pinned LIBERO settling protocol.")
    (output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
