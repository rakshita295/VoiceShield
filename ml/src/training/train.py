"""
Command-Line Interface for Training Custom LFCC-CNN-BiLSTM-Attention Deepfake Detector.

Usage:
    # Run 1-epoch pilot run:
    python -m src.training.train --mode pilot --batch-size 32

    # Run quick smoke test (e.g. 10 batches):
    python -m src.training.train --mode pilot --pilot-train-batches 10 --pilot-val-batches 10

    # Full training:
    python -m src.training.train --mode full --epochs 25 --batch-size 32
"""

import argparse
import sys
import torch

from src.models.lfcc_cnn_bilstm import LFCC_CNN_BiLSTM
from src.datasets.asvspoof2019 import create_asvspoof_dataloader, DEFAULT_DATA_ROOT
from src.training.trainer import Trainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or pilot the custom LFCC-CNN-BiLSTM deepfake detector."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pilot", "full"],
        default="pilot",
        help="Run mode: 'pilot' for 1-epoch benchmark, 'full' for complete training.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Root directory of ASVspoof 2019 LA track.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Total training epochs for full mode.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size per training step.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/custom_lfcc_v1",
        help="Directory to save checkpoints and metrics.",
    )
    parser.add_argument(
        "--use-cmvn",
        action="store_true",
        help="Apply utterance-level Cepstral Mean and Variance Normalization.",
    )
    parser.add_argument(
        "--cmvn-mode",
        type=str,
        choices=["after_deltas", "before_deltas"],
        default="after_deltas",
        help="CMVN application point: 'after_deltas' (recommended) or 'before_deltas'.",
    )
    parser.add_argument(
        "--cmvn-eps",
        type=float,
        default=1e-6,
        help="Epsilon for CMVN numerical stability.",
    )
    parser.add_argument(
        "--pilot-train-batches",
        type=int,
        default=None,
        help="Optional max train batches for pilot smoke testing.",
    )
    parser.add_argument(
        "--pilot-val-batches",
        type=int,
        default=None,
        help="Optional max val batches for pilot smoke testing.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (AMP).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run training on.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (epochs without dev EER improvement).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print(" VoiceShield: Custom LFCC-CNN-BiLSTM-Attention Deepfake Detector")
    print("=" * 70)
    print(f" Mode:             {args.mode.upper()}")
    print(f" Target Device:    {args.device}")
    print(f" Batch Size:       {args.batch_size}")
    print(f" Output Dir:       {args.output_dir}")
    print(f" Mixed Precision:  {not args.no_amp}")
    print("=" * 70)

    # 1. Instantiate DataLoaders
    print("\n[1/4] Loading ASVspoof 2019 LA partitions...")
    train_loader = create_asvspoof_dataloader(
        data_root=args.data_root,
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = create_asvspoof_dataloader(
        data_root=args.data_root,
        split="dev",
        batch_size=args.batch_size,
        shuffle=False,
    )

    class_weights = train_loader.dataset.get_class_weights()
    print(f"  Train trials:    {len(train_loader.dataset):,} (Bonafide: {train_loader.dataset.n_bonafide:,}, Spoof: {train_loader.dataset.n_spoof:,})")
    print(f"  Dev trials:      {len(val_loader.dataset):,} (Bonafide: {val_loader.dataset.n_bonafide:,}, Spoof: {val_loader.dataset.n_spoof:,})")
    print(f"  Class Weights:   Spoof={class_weights[0]:.4f}, Bonafide={class_weights[1]:.4f}")

    # 2. Instantiate Model
    print("\n[2/4] Initializing custom LFCC-CNN-BiLSTM-Attention model...")
    print(f"  CMVN Enabled:    {args.use_cmvn}")
    if args.use_cmvn:
        print(f"  CMVN Mode:       {args.cmvn_mode}")
        print(f"  CMVN Epsilon:    {args.cmvn_eps}")
    model = LFCC_CNN_BiLSTM(
        use_cmvn=args.use_cmvn,
        cmvn_mode=args.cmvn_mode,
        cmvn_eps=args.cmvn_eps,
    )
    param_counts = model.count_parameters()
    print(f"  Total Parameters:     {param_counts['total_parameters']:,}")
    print(f"  Trainable Parameters: {param_counts['trainable_parameters']:,} (100% randomly initialized)")

    # 3. Configure Trainer
    print("\n[3/4] Configuring training engine...")
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.patience,
        mixed_precision=not args.no_amp,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
        use_cmvn=args.use_cmvn,
        cmvn_mode=args.cmvn_mode,
        cmvn_eps=args.cmvn_eps,
    )

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
    )

    if args.resume:
        print(f"  Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"  Resumed at epoch: {start_epoch}")

    # 4. Execute Mode
    if args.mode == "pilot":
        print("\n[4/4] Executing 1-Epoch Pilot Run...")
        summary = trainer.run_pilot(
            max_train_batches=args.pilot_train_batches,
            max_val_batches=args.pilot_val_batches,
        )
        print("\n" + "=" * 70)
        print(" PILOT RUN COMPLETE")
        print("=" * 70)
        print(f" Train Duration:        {summary['train_duration_sec']:.2f} s")
        print(f" Dev Duration:          {summary['val_duration_sec']:.2f} s")
        print(f" Total Epoch Duration:  {summary['total_epoch_duration_sec']:.2f} s (~{summary['total_epoch_duration_sec']/60:.2f} min)")
        print(f" Peak GPU Memory:       {summary['peak_gpu_memory_mb']:.2f} MB ({summary['peak_gpu_memory_gb']:.3f} GB)")
        print(f" Train Loss:            {summary['train_loss']:.5f}")
        print(f" Dev Loss:              {summary['val_loss']:.5f}")
        print(f" Dev EER:               {summary['val_eer_percent']:.3f} %")
        print(f" Operating Tau:         {summary['val_threshold']:.6f}")
        print(f" Dev ROC-AUC:           {summary['val_roc_auc']:.5f}")
        print("=" * 70)
    elif args.mode == "full":
        print(f"\n[4/4] Starting Full Training ({args.epochs} epochs)...")
        history = trainer.fit()
        print("\n" + "=" * 70)
        print(" FULL TRAINING COMPLETE")
        print("=" * 70)
        print(f" Best Dev EER:       {trainer.best_val_eer * 100.0:.3f} %")
        print(f" Best Operating Tau: {trainer.best_threshold:.6f}")
        print("=" * 70)


if __name__ == "__main__":
    main()
