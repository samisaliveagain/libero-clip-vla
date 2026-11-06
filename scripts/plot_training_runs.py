"""Plot recorded training/validation loss without selecting on test data."""
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    runs = [
        ("One task: verified Stage 5", "stage5/put_the_bowl_on_the_plate/baseline"),
        ("Three tasks: failed rollout candidate", "stage6/three_tasks"),
        ("Ten tasks: training in progress", "stage6/ten_tasks_scratch"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, constrained_layout=True)
    for ax, (title, folder) in zip(axes, runs):
        run_dir = ROOT / "outputs" / folder
        rows = []
        for path in sorted(run_dir.glob("training/version_*/metrics.csv")):
            with path.open(newline="") as stream:
                rows.extend(csv.DictReader(stream))
        if "training in progress" in title and (run_dir / "training_result.json").exists():
            title = title.replace("training in progress", "training completed")
        for key, label, color in [("train_loss", "Training", "#176b87"), ("val_loss", "Validation", "#c55a11")]:
            values = [(int(float(r["epoch"])), float(r[key])) for r in rows if r.get(key)]
            if values:
                x, y = zip(*values)
                ax.plot(x, y, label=label, color=color, marker=".", linewidth=1.6)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch (zero-based)")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xlim(left=0, right=max([1]+[int(float(row["epoch"])) for row in rows]))
        ax.grid(alpha=.2)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Pose MSE + gripper MSE")
    fig.suptitle("Offline fit does not establish closed-loop task success", fontsize=14)
    output = ROOT / "outputs/interview/training_curves.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
