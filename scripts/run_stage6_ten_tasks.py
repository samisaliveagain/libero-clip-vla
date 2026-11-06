"""Run the audited ten-task experiment and development checks sequentially.

This runner never marks Stage 6 complete and never accesses the final test set.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/stage6/ten_tasks_scratch"
MANIFEST = ROOT / "data/manifests/libero_goal_stage6.json"


def run(script, args, log_path):
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / script), *map(str, args)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    audit = json.loads((ROOT / "outputs/stage6/data_audit/verification.json").read_text())
    assert audit["status"] == "PASSED" and audit["tasks"] == 10
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT / "training_result.json"
    if not result_path.exists():
        resume = OUTPUT / "checkpoints/last.ckpt"
        resume_args = ["--resume", resume] if resume.exists() else []
        print("RESUME ten-task training" if resume_args else
              "TRAIN ten tasks from fresh policy weights; CLIP remains pretrained and frozen", flush=True)
        run("train_libero_policy.py", ["--mode", "train", "--tasks", "all", "--manifest", MANIFEST,
                                      "--output", OUTPUT, "--epochs", 60, "--patience", 15,
                                      "--microbatch", 32, "--accumulation", 1, "--workers", 1,
                                      "--validation-workers", 0,
                                      "--seed", 20261008, "--cpu-threads", 4, *resume_args], OUTPUT / "train.log")
    checkpoint = json.loads(result_path.read_text())["best_checkpoint"]
    print("EVALUATE", checkpoint, flush=True)
    tasks = [task["task_id"] for task in json.loads(MANIFEST.read_text())["tasks"]]
    common = ["--checkpoint", checkpoint, "--manifest", MANIFEST]
    run("evaluate_development_suite.py", [*common, "--tasks", *tasks, "--modes", "correct",
                                          "--output", OUTPUT / "development"], OUTPUT / "development.log")
    run("evaluate_development_suite.py", [*common, "--modes", "blank", "shuffled",
                                          "--output", OUTPUT / "language"], OUTPUT / "language.log")
    run("evaluate_development_suite.py", [*common, "--modes", "correct", "--reset-indices", 0, 1,
                                          "--scene-task", "put_the_bowl_on_the_plate",
                                          "--output", OUTPUT / "paired_goals"], OUTPUT / "paired_goals.log")
    run("evaluate_policy_offline.py", [*common, "--tasks", *tasks,
                                       "--output", OUTPUT / "validation_errors.json"], OUTPUT / "offline.log")
    (OUTPUT / "development_complete.json").write_text(json.dumps({
        "status": "DEVELOPMENT_CHECKS_FINISHED_REQUIRES_GATE_REVIEW", "checkpoint": checkpoint}, indent=2)+"\n")
    print("Stage 6 development checks finished. Review actual goal achievement before Stage 7.", flush=True)


if __name__ == "__main__":
    main()
