"""Sequential development rollouts; preserve every result and failure video."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOWLS = ["put_the_bowl_on_the_plate", "put_the_bowl_on_the_stove",
         "put_the_bowl_on_top_of_the_cabinet"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/libero_goal_stage2.json")
    parser.add_argument("--tasks", nargs="+", default=BOWLS)
    parser.add_argument("--reset-indices", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--modes", nargs="+", default=["correct", "blank", "shuffled"])
    parser.add_argument("--scene-task", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--settling-steps", type=int, default=5)
    parser.add_argument("--settling-gripper", type=int, choices=[-1, 0], default=0)
    args = parser.parse_args()
    if args.tasks == ["all"]:
        args.tasks = [task["task_id"] for task in json.loads(args.manifest.read_text())["tasks"]]
    reports = []
    for task_index, task in enumerate(args.tasks):
        for mode in args.modes:
            out = args.output / task / mode
            out.mkdir(parents=True, exist_ok=True)
            report = out / "report.json"
            if not report.exists():
                command = [sys.executable, str(ROOT / "scripts/rollout_libero_policy.py"),
                           "--checkpoint", str(args.checkpoint), "--manifest", str(args.manifest),
                           "--task", task, "--instruction-mode", mode, "--output", str(out),
                           "--horizon", str(args.horizon), "--settling-steps", str(args.settling_steps),
                           "--settling-gripper", str(args.settling_gripper),
                           "--reset-indices", *map(str, args.reset_indices)]
                if args.scene_task:
                    command += ["--scene-task", args.scene_task]
                if mode == "shuffled":
                    command += ["--instruction-task", args.tasks[(task_index+1) % len(args.tasks)]]
                print("RUN", task, mode, flush=True)
                with (out / "process.log").open("w", encoding="utf-8") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads(report.read_text())
            assert result["checkpoint_sha256"] == __import__("hashlib").sha256(args.checkpoint.read_bytes()).hexdigest()
            assert result["reset_indices"] == args.reset_indices
            assert result["horizon"] == args.horizon
            assert result["task"] == task and result["instruction_mode"] == mode
            assert result["scene_task"] == (args.scene_task or task)
            assert result["settling_steps"] == args.settling_steps
            assert result.get("settling_gripper", 0) == args.settling_gripper
            reports.append(result)
            print(task, mode, result["success_count"], "/", result["rollout_count"], flush=True)
            (args.output / "suite_report.json").write_text(json.dumps(reports, indent=2)+"\n")


if __name__ == "__main__":
    main()
