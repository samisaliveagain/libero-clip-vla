"""Continue frozen-checkpoint evaluation serially on the local GPU.

Training already stopped successfully. Coverage is extended after the original
two-task results were seen. Remaining Stage 7 robustness checks are separate.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "outputs/stage7/frozen_epoch06.ckpt"
MANIFEST = ROOT / "data/manifests/libero_goal_stage6.json"
EXPECTED_HASH = "3731f123462be5bee742cfcb45a595386b68a5963150e1a7079328315eca36cf"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != EXPECTED_HASH:
        raise ValueError("Frozen checkpoint hash changed")
    tasks = [task["task_id"] for task in json.loads(MANIFEST.read_text())["tasks"]]
    completed_tasks = ["put_the_bowl_on_the_plate", "put_the_bowl_on_top_of_the_cabinet"]
    plan = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": EXPECTED_HASH,
        "manifest": str(MANIFEST), "offline_split": "test", "offline_tasks": tasks,
        "additional_rollout_tasks": [task for task in tasks if task not in completed_tasks],
        "existing_completed_tasks": completed_tasks, "reset_indices": list(range(20, 40)),
        "horizon": 600, "seed_base": 20261007,
        "settling_steps": 10, "settling_gripper": -1,
        "selection_note": "Task coverage extended after original cabinet/plate results were known. No checkpoint reselection or training.",
        "stage7_gate": "Input ablations, shifted-condition evaluation and unwanted-contact measures remain separate; job completion does not close the Stage 7 gate.",
    }
    if (output / "plan.json").exists():
        old = json.loads((output / "plan.json").read_text())
        if {k: v for k, v in old.items() if k != "created_at_utc"} != {k: v for k, v in plan.items() if k != "created_at_utc"}:
            raise ValueError("Existing plan differs")
    else:
        write_json(output / "plan.json", plan)
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    common = ["--checkpoint", str(CHECKPOINT), "--manifest", str(MANIFEST)]
    jobs = [("offline_test", "evaluate_policy_offline.py",
             common + ["--tasks", *tasks, "--split", "test", "--output", str(output / "offline_test.json")],
             output / "offline_test.json")]
    for task in plan["additional_rollout_tasks"]:
        destination = output / "heldout" / task / "correct"
        jobs.append((task, "rollout_libero_policy.py", common + [
            "--task", task, "--reset-indices", *map(str, plan["reset_indices"]),
            "--horizon", "600", "--seed", "20261007", "--settling-steps", "10",
            "--settling-gripper", "-1", "--output", str(destination)], destination / "report.json"))
    status = {"pid": os.getpid(), "state": "starting", "completed_jobs": [], "total_jobs": len(jobs)}
    keep_awake = None
    if os.name == "nt":
        keep_awake = ctypes.windll.kernel32.SetThreadExecutionState
        status["temporary_sleep_prevention"] = bool(keep_awake(0x80000001))
    try:
        for name, script, arguments, report_path in jobs:
            status.update(state="running", current_job=name, child_pid=None)
            write_json(output / "status.json", status)
            if not report_path.exists():
                with (output / f"{name}.log").open("a", encoding="utf-8") as log:
                    child = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts" / script), *arguments],
                                             cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    status["child_pid"] = child.pid
                    write_json(output / "status.json", status)
                    print(f"Started {name}, PID {child.pid}", flush=True)
                    code = child.wait()
                if code:
                    raise RuntimeError(f"{name} exited {code}; see {output / (name + '.log')}")
            report = json.loads(report_path.read_text())
            if report["checkpoint_sha256"] != EXPECTED_HASH:
                raise ValueError(f"Checkpoint mismatch in {name}")
            if name != "offline_test":
                if report["reset_indices"] != plan["reset_indices"] or report["rollout_count"] != 20:
                    raise ValueError(f"Incomplete reset coverage in {name}")
                faults = [r for r in report["results"] if r["stop_reason"] not in ("horizon", "success")]
                if faults:
                    raise RuntimeError(f"{name} recorded execution faults: {faults}")
            status["completed_jobs"].append(name)
            print(f"Completed {name}", flush=True)
        status.update(state="complete", current_job=None, child_pid=None,
                      finished_at_utc=datetime.now(timezone.utc).isoformat(), stage7_gate="open")
        write_json(output / "completion.json", status)
    except Exception:
        status.update(state="failed", error=traceback.format_exc())
        print(status["error"], flush=True)
        raise
    finally:
        if keep_awake is not None:
            keep_awake(0x80000000)
        write_json(output / "status.json", status)


if __name__ == "__main__":
    main()
