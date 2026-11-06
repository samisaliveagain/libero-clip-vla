"""Stage 5 one-task diagnostic and Lightning behavior cloning.

Windows CMD examples:
    venv\Scripts\python.exe scripts\train_libero_policy.py --mode diagnostic
    venv\Scripts\python.exe scripts\train_libero_policy.py --mode train
"""

from __future__ import annotations

import argparse
import json
import os
import hashlib
import time
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "mplconfig"))

import lightning.pytorch as pl  # noqa: E402
import torch  # noqa: E402
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint  # noqa: E402
from lightning.pytorch.loggers import CSVLogger  # noqa: E402
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler  # noqa: E402

from clip_features import FrozenCLIPFeatures  # noqa: E402
from libero_dataset_adapter import AlignedLiberoSequenceDataset, DEFAULT_MANIFEST  # noqa: E402
from libero_policy import LiberoActionPolicy, behavior_cloning_loss  # noqa: E402


TASK = "put_the_bowl_on_the_plate"
OUTPUT = ROOT / "outputs" / "stage5" / TASK


def initialize_data_worker(worker_id):
    # Windows spawn starts a fresh interpreter; robomimic's process-global
    # observation registry must be initialized in each worker as well.
    import robomimic.utils.obs_utils as obs_utils
    obs_utils.initialize_obs_modality_mapping_from_dict({
        "rgb": ["agentview_rgb"], "low_dim": ["joint_states", "gripper_states"]})


class EpochProgress(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.started = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        print(json.dumps({"epoch": trainer.current_epoch,
                          "seconds": round(time.perf_counter() - self.started, 2),
                          "val_loss": float(trainer.callback_metrics["val_loss"]),
                          "peak_allocated_mib": round(torch.cuda.max_memory_allocated()/2**20, 2)}), flush=True)


class BehaviorCloningModule(pl.LightningModule):
    """Frozen CLIP feature extraction plus the learned robot policy."""

    def __init__(self, learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.features = FrozenCLIPFeatures(device="cuda")
        self.policy = LiberoActionPolicy()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # The established CLIP processor consumes the Stage 2 CPU RGB array.
        # Move only small state/action tensors; it moves processed pixels itself.
        batch["proprio"] = batch["proprio"].to(device, non_blocking=True)
        batch["action"] = batch["action"].to(device, non_blocking=True)
        return batch

    def forward(self, batch):
        visual = self.features.encode_images(batch["image"])
        text, mask = self.features.encode_text(batch["instruction"])
        return self.policy(visual, text, mask, batch["proprio"])

    def _step(self, batch, split: str):
        prediction = self(batch)
        target = batch["action"]
        loss = behavior_cloning_loss(prediction, target)
        self.log(f"{split}_loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=len(target))
        self.log(f"{split}_translation_mse", (prediction[:, :3] - target[:, :3]).square().mean(),
                 on_step=False, on_epoch=True, batch_size=len(target))
        self.log(f"{split}_rotation_mse", (prediction[:, 3:6] - target[:, 3:6]).square().mean(),
                 on_step=False, on_epoch=True, batch_size=len(target))
        self.log(f"{split}_gripper_mse", (prediction[:, 6] - target[:, 6]).square().mean(),
                 on_step=False, on_epoch=True, batch_size=len(target))
        if split == "val":
            task_ids = batch["metadata"]["task_id"]
            for task in set(task_ids):
                selected = torch.tensor([t == task for t in task_ids], device=target.device)
                self.log(f"val_task/{task}", behavior_cloning_loss(prediction[selected], target[selected]),
                         on_step=False, on_epoch=True, batch_size=int(selected.sum()))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.policy.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4
        )


def diagnostic(steps: int = 150) -> None:
    """Show that eight fixed training examples can be memorized without dropout."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset = AlignedLiberoSequenceDataset(DEFAULT_MANIFEST, TASK, "train")
    try:
        batch = next(iter(DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)))
    finally:
        dataset.close()
    features = FrozenCLIPFeatures()
    policy = LiberoActionPolicy().to(features.device)
    policy.eval()  # Disables dropout, but leaves autograd enabled.
    visual = features.encode_images(batch["image"])
    text, mask = features.encode_text(batch["instruction"])
    proprio = batch["proprio"].to(features.device)
    target = batch["action"].to(features.device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-4)
    initial = None
    losses = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = policy(visual, text, mask, proprio)
        loss = behavior_cloning_loss(prediction, target)
        if initial is None:
            initial = float(loss.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 25 == 0:
            print(f"diagnostic step {step+1}/{steps}: loss={losses[-1]:.6f}", flush=True)
    passed = losses[-1] < initial * 0.25
    report = {
        "status": "PASSED" if passed else "FAILED",
        "fixed_train_examples": 8,
        "dropout_enabled": False,
        "steps": steps,
        "initial_loss": initial,
        "final_loss": losses[-1],
        "final_to_initial_ratio": losses[-1] / initial,
        "criterion": "final loss < 25% of initial loss",
    }
    (OUTPUT / "tiny_overfit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise RuntimeError("Tiny-set diagnostic did not pass; do not start long training")


def train(args) -> None:
    report_path = OUTPUT / "tiny_overfit.json"
    if not report_path.exists() or json.loads(report_path.read_text())["status"] != "PASSED":
        raise RuntimeError("Run and pass --mode diagnostic before full training")
    run_dir = args.output or OUTPUT / args.run_name
    if (run_dir / "run_config.json").exists() and args.resume is None:
        raise FileExistsError(f"Use a new output directory or --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and (run_dir / "run_config.json").exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy2(run_dir / "run_config.json", run_dir / f"run_config_before_resume_{stamp}.json")
    train_sets = [AlignedLiberoSequenceDataset(args.manifest, task, "train") for task in args.tasks]
    val_sets = [AlignedLiberoSequenceDataset(args.manifest, task, "validation") for task in args.tasks]
    train_data, val_data = ConcatDataset(train_sets), ConcatDataset(val_sets)
    # Each task, then each episode, then each valid row has equal conditional
    # sampling probability. Long demonstrations do not dominate the objective.
    weights = []
    for dataset in train_sets:
        for index in range(len(dataset)):
            episode = dataset.reader._index_to_demo_id[index]
            rows = dataset.reader._demo_id_to_demo_length[episode] - 1
            weights.append(1.0 / (len(train_sets) * len(dataset.demos) * rows))
    sampler = WeightedRandomSampler(weights, num_samples=len(train_data), replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    config = {
        "tasks": args.tasks, "epochs": args.epochs, "microbatch": args.microbatch,
        "accumulation": args.accumulation, "effective_batch": args.microbatch * args.accumulation,
        "learning_rate": args.learning_rate, "weight_decay": 1e-4, "gradient_clip_norm": 1.0,
        "precision": "bf16-mixed", "train_examples": len(train_data),
        "validation_examples": len(val_data),
        "checkpoint_revision": "57c216476eefef5ab752ec549e440a49ae4ae5f3",
        "manifest": str(args.manifest.resolve()), "seed": args.seed,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "sampler": "uniform task -> uniform episode -> uniform valid row",
        "initial_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "resume_checkpoint": str(args.resume) if args.resume else None,
        "dataloader_workers": args.workers,
        "validation_workers": args.validation_workers,
        "cpu_threads": args.cpu_threads,
        "resume_note": "Lightning restores epoch/weights/optimizer; a resume is not claimed bitwise identical" if args.resume else None,
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    train_loader = DataLoader(
        train_data, batch_size=args.microbatch, sampler=sampler, num_workers=args.workers,
        persistent_workers=args.workers > 0, pin_memory=True, worker_init_fn=initialize_data_worker
    )
    val_loader = DataLoader(
        val_data, batch_size=args.microbatch, shuffle=False, num_workers=args.validation_workers,
        persistent_workers=args.validation_workers > 0, pin_memory=True, worker_init_fn=initialize_data_worker
    )
    module = BehaviorCloningModule(args.learning_rate)
    if args.init_checkpoint:
        checkpoint_data = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        module.load_state_dict(checkpoint_data["state_dict"], strict=True)
    torch.set_float32_matmul_precision("high")
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints", filename="epoch{epoch:02d}-val{val_loss:.4f}",
        monitor="val_loss", mode="min", save_top_k=3, save_last=True,
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision="bf16-mixed", max_epochs=args.epochs,
        accumulate_grad_batches=args.accumulation, gradient_clip_val=1.0,
        callbacks=[checkpoint, EarlyStopping(monitor="val_loss", mode="min", patience=args.patience), EpochProgress()],
        logger=CSVLogger(save_dir=str(run_dir), name="training"),
        log_every_n_steps=10, num_sanity_val_steps=0,
        enable_progress_bar=False,
    )
    try:
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader,
                    ckpt_path=str(args.resume) if args.resume else None)
        print("BEST_CHECKPOINT", checkpoint.best_model_path, flush=True)
        print("BEST_VALIDATION_LOSS", float(checkpoint.best_model_score), flush=True)
        (run_dir / "training_result.json").write_text(json.dumps({
            "best_checkpoint": checkpoint.best_model_path,
            "best_validation_loss": float(checkpoint.best_model_score),
            "epochs_completed": trainer.current_epoch,
            "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
        }, indent=2)+"\n", encoding="utf-8")
    except BaseException:
        try:
            trainer.save_checkpoint(run_dir / "interrupted.ckpt")
        except Exception:
            pass  # Preserve the original error if checkpointing also fails.
        raise
    finally:
        for dataset in train_sets + val_sets:
            dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("diagnostic", "train"), required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--tasks", nargs="+", default=[TASK])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--validation-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.tasks == ["all"]:
        args.tasks = [task["task_id"] for task in json.loads(args.manifest.read_text())["tasks"]]
    if args.init_checkpoint and args.resume:
        parser.error("--init-checkpoint and --resume are mutually exclusive")
    pl.seed_everything(args.seed, workers=True)
    torch.set_num_threads(args.cpu_threads)
    if args.mode == "diagnostic":
        diagnostic()
    else:
        train(args)


if __name__ == "__main__":
    main()
