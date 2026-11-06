"""Evaluate a frozen development candidate under an explicit startup protocol."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/libero_goal_stage6.json")
    parser.add_argument("--settling-steps", type=int, default=10)
    parser.add_argument("--settling-gripper", type=int, default=-1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    common = ["--checkpoint", str(args.checkpoint), "--manifest", str(args.manifest),
              "--settling-steps", str(args.settling_steps), "--settling-gripper", str(args.settling_gripper)]
    jobs = [
        ("all_tasks", ["--tasks", "all", "--modes", "correct"]),
        ("language", ["--modes", "blank", "shuffled"]),
        ("paired_goals", ["--modes", "correct", "--scene-task", "put_the_bowl_on_the_plate"]),
    ]
    for name, extra in jobs:
        print("EVALUATE", name, flush=True)
        with (args.output / f"{name}.log").open("a", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-u", str(ROOT / "scripts/evaluate_development_suite.py"),
                            *common, *extra, "--output", str(args.output / name)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    (args.output / "complete.json").write_text(json.dumps({
        "status": "DEVELOPMENT_CHECKS_FINISHED_REQUIRES_GATE_REVIEW",
        "checkpoint": str(args.checkpoint), "settling_steps": args.settling_steps,
        "settling_gripper": args.settling_gripper}, indent=2)+"\n")
    print("Candidate development checks completed; review the gate.", flush=True)


if __name__ == "__main__":
    main()
