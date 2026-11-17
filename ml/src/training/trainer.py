"""
Training Engine for Custom LFCC-CNN-BiLSTM-Attention Deepfake Speech Detector.

Features:
- Configurable hyperparameters via Dataclass / Dict.
- Mixed precision training (torch.amp.autocast + GradScaler).
- Class-weighted Cross-Entropy Loss to balance ~1:8.8 bonafide-to-spoof imbalance.
- AdamW optimizer with CosineAnnealingLR scheduling.
- Gradient clipping for recurrent LSTM stability.
- Validation computing Equal Error Rate (EER), operating threshold tau, and ROC-AUC.
- Checkpointing best model based strictly on development partition EER.
- Checkpoint resume capability.
- Early stopping support.
- Pilot smoke mode for measuring exact GPU VRAM footprint and epoch latency.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List, Tuple
import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.lfcc_cnn_bilstm import LFCC_CNN_BiLSTM
from src.evaluation.metrics import (
    compute_eer,
    compute_roc_auc,
    evaluate_at_threshold,
)


@dataclass
class TrainingConfig:
    epochs: int = 25
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 5
    mixed_precision: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    output_dir: str = "experiments/custom_lfcc_v1"
    class_weighted_loss: bool = True
    max_duration_sec: float = 4.0
    save_every_epoch: bool = True
    use_cmvn: bool = False
    cmvn_mode: str = "after_deltas"
    cmvn_eps: float = 1e-6


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seed across Python, NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    """
    Manages the training, validation, metric evaluation, and checkpointing.
    """

    def __init__(
        self,
        model: LFCC_CNN_BiLSTM,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        set_seed(self.config.seed)

        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)

        # Loss function
        if self.config.class_weighted_loss and class_weights is not None:
            weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer and Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=1e-6,
        )

        # Mixed precision GradScaler
        self.use_amp = self.config.mixed_precision and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # State tracking
        self.start_epoch = 1
        self.best_val_eer = float("inf")
        self.best_threshold = 0.0
        self.patience_counter = 0
        self.history: List[Dict[str, Any]] = []

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Save initial configuration
        config_path = os.path.join(self.config.output_dir, "training_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)

    def train_epoch(
        self,
        epoch: int,
        max_batches: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Runs one training epoch.

        Returns:
            Tuple of (average_loss, epoch_duration_seconds).
        """
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        start_time = time.time()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        total_batches = len(self.train_loader) if max_batches is None else min(len(self.train_loader), max_batches)
        
        for batch_idx, (audio, labels, lengths, _) in enumerate(self.train_loader, start=1):
            audio = audio.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            lengths = lengths.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits, _ = self.model(audio, audio_lengths=lengths)
                loss = self.criterion(logits, labels)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

            total_loss += float(loss.item())
            batch_count += 1

            if batch_idx % 50 == 0 or batch_idx == total_batches:
                elapsed = time.time() - start_time
                avg_b_loss = total_loss / batch_count
                samples_per_sec = (batch_count * self.config.batch_size) / max(1e-3, elapsed)
                vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2) if torch.cuda.is_available() else 0.0
                print(f"  [Train Epoch {epoch}] Batch {batch_idx:3d}/{total_batches:3d} | Loss: {avg_b_loss:.4f} | {samples_per_sec:.1f} samples/s | Peak VRAM: {vram_mb:.1f} MB", flush=True)

            if max_batches is not None and batch_count >= max_batches:
                break

        duration = time.time() - start_time
        avg_loss = total_loss / max(1, batch_count)
        return avg_loss, duration

    @torch.no_grad()
    def validate(
        self,
        max_batches: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluates model on validation (development) split.

        Computes:
        - Val loss
        - Equal Error Rate (EER) and operating threshold tau
        - ROC-AUC
        - Precision, Recall, Specificity, F1, FAR, and FRR at tau
        """
        self.model.eval()
        total_loss = 0.0
        batch_count = 0

        target_scores_list: List[float] = []
        nontarget_scores_list: List[float] = []

        total_batches = len(self.val_loader) if max_batches is None else min(len(self.val_loader), max_batches)
        start_val_time = time.time()

        for batch_idx, (audio, labels, lengths, _) in enumerate(self.val_loader, start=1):
            audio = audio.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            lengths = lengths.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits, _ = self.model(audio, audio_lengths=lengths)
                loss = self.criterion(logits, labels)

            total_loss += float(loss.item())
            batch_count += 1

            # Score = bonafide_logit (index 1) - spoof_logit (index 0)
            diff_scores = (logits[:, 1] - logits[:, 0]).cpu().numpy()
            labels_np = labels.cpu().numpy()

            for score_val, lbl in zip(diff_scores, labels_np):
                if lbl == 1:
                    target_scores_list.append(float(score_val))
                else:
                    nontarget_scores_list.append(float(score_val))

            if batch_idx % 100 == 0 or batch_idx == total_batches:
                elapsed = time.time() - start_val_time
                samples_sec = (batch_count * self.config.batch_size) / max(1e-3, elapsed)
                print(f"  [Validation] Batch {batch_idx:3d}/{total_batches:3d} | Evaluated {len(target_scores_list) + len(nontarget_scores_list):,} trials | {samples_sec:.1f} samples/s", flush=True)

            if max_batches is not None and batch_count >= max_batches:
                break

        avg_loss = total_loss / max(1, batch_count)
        target_arr = np.array(target_scores_list, dtype=np.float64)
        nontarget_arr = np.array(nontarget_scores_list, dtype=np.float64)

        if len(target_arr) > 0 and len(nontarget_arr) > 0:
            eer, threshold = compute_eer(target_arr, nontarget_arr)
            auc = compute_roc_auc(target_arr, nontarget_arr)
            threshold_metrics = evaluate_at_threshold(target_arr, nontarget_arr, threshold)
        else:
            eer, threshold, auc = 0.5, 0.0, 0.5
            threshold_metrics = {}

        return {
            "val_loss": round(avg_loss, 5),
            "val_eer": round(float(eer), 5),
            "val_eer_percent": round(float(eer) * 100.0, 3),
            "val_threshold": round(float(threshold), 6),
            "val_roc_auc": round(float(auc), 5),
            "detailed_metrics": threshold_metrics,
            "target_count": len(target_arr),
            "nontarget_count": len(nontarget_arr),
        }

    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool,
        val_metrics: Dict[str, Any]
    ) -> str:
        """Saves latest, per-epoch, and optionally best checkpoint."""
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "val_metrics": val_metrics,
            "config": asdict(self.config),
            "best_val_eer": self.best_val_eer,
            "best_threshold": self.best_threshold,
        }

        latest_path = os.path.join(self.config.output_dir, "latest_checkpoint.pt")
        torch.save(checkpoint_data, latest_path)

        if self.config.save_every_epoch:
            epoch_path = os.path.join(self.config.output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(checkpoint_data, epoch_path)

        if is_best:
            best_path = os.path.join(self.config.output_dir, "best_model.pt")
            torch.save(checkpoint_data, best_path)
            return best_path
        return latest_path

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Loads model and optimizer states to resume training."""
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.use_amp and checkpoint.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_eer = checkpoint.get("best_val_eer", float("inf"))
        self.best_threshold = checkpoint.get("best_threshold", 0.0)
        return self.start_epoch

    def fit(self) -> List[Dict[str, Any]]:
        """
        Executes full training across config.epochs with early stopping and checkpointing.
        """
        print(f"\nStarting training from epoch {self.start_epoch} to {self.config.epochs}...")
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            train_loss, train_duration = self.train_epoch(epoch)
            val_start = time.time()
            val_metrics = self.validate()
            val_duration = time.time() - val_start

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            val_eer = val_metrics["val_eer"]
            is_best = val_eer < self.best_val_eer

            if is_best:
                self.best_val_eer = val_eer
                self.best_threshold = val_metrics["val_threshold"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch=epoch, is_best=is_best, val_metrics=val_metrics)

            detail = val_metrics.get("detailed_metrics", {})
            peak_vram = (
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                if torch.cuda.is_available()
                else 0.0
            )

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": val_metrics["val_loss"],
                "val_eer": val_metrics["val_eer"],
                "val_eer_percent": val_metrics["val_eer_percent"],
                "val_threshold": val_metrics["val_threshold"],
                "val_roc_auc": val_metrics["val_roc_auc"],
                "val_accuracy": detail.get("accuracy"),
                "val_precision": detail.get("precision"),
                "val_recall": detail.get("recall_sensitivity"),
                "val_specificity": detail.get("specificity"),
                "val_f1": detail.get("f1_score"),
                "val_far": detail.get("false_acceptance_rate"),
                "val_frr": detail.get("false_rejection_rate"),
                "learning_rate": current_lr,
                "train_duration_sec": round(train_duration, 2),
                "val_duration_sec": round(val_duration, 2),
                "peak_vram_mb": round(peak_vram, 2),
                "is_best": is_best,
            }
            self.history.append(epoch_record)

            history_path = os.path.join(self.config.output_dir, "training_history.json")
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)

            best_marker = " [BEST]" if is_best else ""
            print(
                f"--> [Epoch {epoch:2d}/{self.config.epochs:2d}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Dev Loss: {val_metrics['val_loss']:.4f} | "
                f"Dev EER: {val_metrics['val_eer_percent']:.2f}% | "
                f"AUC: {val_metrics['val_roc_auc']:.4f} | "
                f"Acc: {detail.get('accuracy', 0.0)*100:.2f}% | "
                f"LR: {current_lr:.2e}{best_marker}",
                flush=True,
            )

            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\n[Early Stopping] Dev EER has not improved for {self.patience_counter} epochs. Stopping.")
                break

        print(f"\nTraining completed. Best Dev EER: {self.best_val_eer * 100.0:.3f}% at threshold tau: {self.best_threshold:.6f}")
        return self.history

    def run_pilot(
        self,
        max_train_batches: Optional[int] = None,
        max_val_batches: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes a 1-epoch pilot run to measure throughput, memory, and loss.

        Args:
            max_train_batches: If set, caps the number of train batches for quick smoke test.
            max_val_batches: If set, caps the number of val batches.

        Returns:
            Dictionary containing benchmark metrics.
        """
        peak_vram_before = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            peak_vram_before = torch.cuda.memory_allocated(self.device) / (1024 ** 2)

        train_loss, train_duration = self.train_epoch(epoch=1, max_batches=max_train_batches)

        peak_vram_mb = 0.0
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

        val_start = time.time()
        val_metrics = self.validate(max_batches=max_val_batches)
        val_duration = time.time() - val_start

        self.best_val_eer = val_metrics["val_eer"]
        self.best_threshold = val_metrics["val_threshold"]
        self.save_checkpoint(epoch=1, is_best=True, val_metrics=val_metrics)

        summary = {
            "epoch": 1,
            "train_loss": round(train_loss, 5),
            "train_duration_sec": round(train_duration, 2),
            "val_duration_sec": round(val_duration, 2),
            "total_epoch_duration_sec": round(train_duration + val_duration, 2),
            "peak_gpu_memory_mb": round(peak_vram_mb, 2),
            "peak_gpu_memory_gb": round(peak_vram_mb / 1024.0, 3),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "mixed_precision": self.use_amp,
            "val_loss": val_metrics["val_loss"],
            "val_eer_percent": val_metrics["val_eer_percent"],
            "val_threshold": val_metrics["val_threshold"],
            "val_roc_auc": val_metrics["val_roc_auc"],
        }

        # Save pilot summary
        summary_path = os.path.join(self.config.output_dir, "pilot_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
