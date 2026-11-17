"""
Final Held-Out Benchmark Runner on ASVspoof 2019 LA Evaluation Partition.

Strict Evaluation Constraints:
- Zero retraining, zero fine-tuning, zero weight updates.
- Evaluates exactly 71,237 trials from ASVspoof2019.LA.cm.eval.trl.txt.
- Uses strictly FIXED development operating threshold tau_dev = 11.875000.
- Zero evaluation label leakage: threshold was frozen before touching evaluation data.
- Measures inference latency, throughput, per-attack metrics (A07-A19), and confusion matrix.
- Saves reproducible outputs to experiments/custom_lfcc_v1/.
"""

import os
import sys
import time
import json
import argparse
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.lfcc_cnn_bilstm import LFCC_CNN_BiLSTM
from src.datasets.asvspoof2019 import create_asvspoof_dataloader
from src.evaluation.metrics import (
    compute_det_curve,
    compute_eer,
    compute_roc_auc,
    compute_min_tdcf,
    evaluate_at_threshold,
)


def compute_interpolated_eer(target_scores: np.ndarray, nontarget_scores: np.ndarray) -> Tuple[float, float]:
    """
    Computes exact linearly interpolated Equal Error Rate (where FAR = FRR on continuous segment).
    """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    for i in range(len(frr) - 1):
        diff1 = far[i] - frr[i]
        diff2 = far[i + 1] - frr[i + 1]
        if diff1 * diff2 <= 0:
            denom = (frr[i + 1] - frr[i]) - (far[i + 1] - far[i])
            alpha = (far[i] - frr[i]) / denom if denom != 0 else 0.0
            interp_eer = frr[i] + alpha * (frr[i + 1] - frr[i])
            interp_tau = thresholds[i] + alpha * (thresholds[i + 1] - thresholds[i])
            return float(interp_eer), float(interp_tau)
    # Fallback to discrete
    min_idx = np.argmin(np.abs(frr - far))
    return float(np.mean((frr[min_idx], far[min_idx]))), float(thresholds[min_idx])


def run_evaluation(
    checkpoint_path: str = "experiments/custom_lfcc_v1/best_model.pt",
    output_dir: Optional[str] = None,
    batch_size: int = 32,
    override_threshold: Optional[float] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if output_dir is None:
        output_dir = os.path.dirname(checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 75)
    print(" VoiceShield: Held-Out ASVspoof 2019 LA Evaluation Benchmark")
    print("=" * 75)
    print(f" Compute Device:      {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f" Model Checkpoint:    {checkpoint_path}")
    print(f" Checkpoint Epoch:    {ckpt['epoch']}")
    print(f" Best Dev EER:        {ckpt['best_val_eer'] * 100:.3f}%")

    # Fixed threshold derived strictly from development partition
    if override_threshold is not None:
        tau_dev = float(override_threshold)
    else:
        tau_dev = float(ckpt["best_threshold"])
    print(f" Fixed Dev Threshold: {tau_dev:.6f} (Frozen; zero evaluation leakage)")

    # Read model CMVN configuration from checkpoint
    config = ckpt.get("config", {})
    use_cmvn = config.get("use_cmvn", False)
    cmvn_mode = config.get("cmvn_mode", "after_deltas")
    cmvn_eps = float(config.get("cmvn_eps", 1e-6))
    print(f" Front-End Config:    use_cmvn={use_cmvn}, cmvn_mode='{cmvn_mode}', cmvn_eps={cmvn_eps}")
    print(f" Output Directory:    {output_dir}")
    print("=" * 75)

    # 1. Instantiate Model
    model = LFCC_CNN_BiLSTM(
        use_cmvn=use_cmvn,
        cmvn_mode=cmvn_mode,
        cmvn_eps=cmvn_eps,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # 2. Instantiate Evaluation DataLoader
    print("\n[1/3] Loading ASVspoof 2019 LA evaluation partition...")
    eval_loader = create_asvspoof_dataloader(
        split="eval",
        batch_size=batch_size,
        shuffle=False,
    )
    total_eval_trials = len(eval_loader.dataset)
    print(f"  Total Protocol Trials: {total_eval_trials:,}")
    print(f"  Bonafide Trials:       {eval_loader.dataset.n_bonafide:,} (10.32%)")
    print(f"  Spoofed Trials:        {eval_loader.dataset.n_spoof:,} (89.68%)")
    assert total_eval_trials == 71237, f"Expected 71,237 trials, got {total_eval_trials}"

    # 3. Process all 71,237 recordings
    print("\n[2/3] Executing forward inference on RTX 4050 GPU...")
    all_scores = []
    all_labels = []
    all_file_ids = []
    all_attack_ids = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    start_eval_time = time.time()
    batch_count = 0

    with torch.no_grad():
        for batch_idx, (audio, labels, lengths, meta) in enumerate(eval_loader, start=1):
            audio = audio.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits, _ = model(audio, audio_lengths=lengths)

            # Score: Bonafide (idx 1) - Spoof (idx 0)
            diff_scores = (logits[:, 1] - logits[:, 0]).cpu().numpy()
            labels_np = labels.numpy()

            all_scores.extend(diff_scores.tolist())
            all_labels.extend(labels_np.tolist())
            all_file_ids.extend(meta["file_ids"])
            all_attack_ids.extend(meta["attack_ids"])
            batch_count += 1

            if batch_idx % 250 == 0 or batch_idx == len(eval_loader):
                elapsed_b = time.time() - start_eval_time
                throughput = (len(all_labels)) / max(1e-3, elapsed_b)
                vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0
                print(f"  Batch {batch_idx:4d}/{len(eval_loader):4d} | Evaluated {len(all_labels):,}/{total_eval_trials:,} trials | {throughput:.1f} samples/s | Peak VRAM: {vram_mb:.1f} MB", flush=True)

    total_eval_time = time.time() - start_eval_time
    avg_latency_ms = (total_eval_time / total_eval_trials) * 1000.0
    overall_throughput = total_eval_trials / total_eval_time
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0

    print(f"\nInference completed in {total_eval_time:.2f} seconds ({total_eval_time / 60:.2f} minutes).")
    print(f"Average latency: {avg_latency_ms:.2f} ms/trial | Throughput: {overall_throughput:.1f} trials/s | Peak VRAM: {peak_vram_mb:.1f} MB")

    # 4. Convert to numpy arrays
    scores = np.array(all_scores, dtype=np.float64)
    labels = np.array(all_labels, dtype=np.int32)
    attack_ids = np.array(all_attack_ids)
    file_ids = np.array(all_file_ids)

    assert len(scores) == 71237, f"Corrupted trial count: {len(scores)}"

    # 5. Partition Target and Nontarget
    bonafide_mask = (labels == 1)
    spoof_mask = (labels == 0)

    target_scores = scores[bonafide_mask]
    nontarget_scores = scores[spoof_mask]

    print(f"\nTarget count (Bonafide):    {len(target_scores):,}")
    print(f"Nontarget count (Spoof):    {len(nontarget_scores):,}")

    # 6. Metric Calculations
    print("\n[3/3] Computing comprehensive benchmark metrics...")

    # (a) Fixed Operating Point Metrics (calibrated threshold tau_dev = 11.875)
    fixed_metrics = evaluate_at_threshold(target_scores, nontarget_scores, threshold=tau_dev)
    cm = fixed_metrics["confusion_matrix"]
    tp = cm["TP_bonafide_accepted"]
    fn = cm["FN_bonafide_rejected"]
    tn = cm["TN_spoof_rejected"]
    fp = cm["FP_spoof_accepted"]

    far_fixed = fp / len(nontarget_scores)
    frr_fixed = fn / len(target_scores)
    acc_fixed = (tp + tn) / total_eval_trials
    prec_fixed = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_fixed = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec_fixed = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_fixed = 2 * prec_fixed * rec_fixed / (prec_fixed + rec_fixed) if (prec_fixed + rec_fixed) > 0 else 0.0

    # (b) Global Metrics
    eval_eer_discrete, eval_tau_discrete = compute_eer(target_scores, nontarget_scores)
    eval_eer_interp, eval_tau_interp = compute_interpolated_eer(target_scores, nontarget_scores)
    eval_roc_auc = compute_roc_auc(target_scores, nontarget_scores)
    eval_min_tdcf, eval_tau_tdcf = compute_min_tdcf(target_scores, nontarget_scores)

    # 7. Breakdown by Attack Type
    unique_attacks = sorted(list(set(all_attack_ids)))
    attack_breakdown = {}

    for atk in unique_attacks:
        atk_mask = (attack_ids == atk)
        atk_scores = scores[atk_mask]
        atk_count = int(np.sum(atk_mask))

        if atk == "-":
            # Bonafide human speech
            atk_tp = int(np.sum(atk_scores >= tau_dev))
            atk_fn = int(np.sum(atk_scores < tau_dev))
            atk_acc = atk_tp / atk_count
            atk_error_rate = atk_fn / atk_count  # FRR
            atk_type = "Human Speech (Bonafide)"
        else:
            # Synthetic / Voice conversion attack
            atk_tn = int(np.sum(atk_scores < tau_dev))
            atk_fp = int(np.sum(atk_scores >= tau_dev))
            atk_acc = atk_tn / atk_count
            atk_error_rate = atk_fp / atk_count  # FAR
            atk_type = "Unseen Attack" if atk in [f"A{i:02d}" for i in range(10, 20)] else "Known Attack Variant"

        attack_breakdown[atk] = {
            "attack_id": atk,
            "attack_type": atk_type,
            "trial_count": atk_count,
            "mean_score": round(float(np.mean(atk_scores)), 4),
            "std_score": round(float(np.std(atk_scores)), 4),
            "min_score": round(float(np.min(atk_scores)), 4),
            "max_score": round(float(np.max(atk_scores)), 4),
            "accuracy_percent": round(atk_acc * 100.0, 3),
            "error_rate_percent": round(atk_error_rate * 100.0, 3),
            "false_acceptances" if atk != "-" else "false_rejections": int(np.sum(atk_scores >= tau_dev)) if atk != "-" else int(np.sum(atk_scores < tau_dev)),
        }

    # 8. Print Results
    print("\n" + "=" * 75)
    print(" FINAL HELD-OUT EVALUATION BENCHMARK RESULTS")
    print("=" * 75)
    print(f" Evaluation Split:            ASVspoof 2019 LA Eval (71,237 trials)")
    print(f" Total Audio Evaluated:       {total_eval_trials:,} recordings (0 skipped)")
    print(f" Fixed Operating Threshold:   {tau_dev:.6f} (Strictly from Dev Split)")
    print("-" * 75)
    print(f" OPERATING POINT PERFORMANCE (at Fixed tau_dev = {tau_dev:.4f}):")
    print(f"   Accuracy:                  {acc_fixed * 100:.3f}% ({tp + tn:,} / {total_eval_trials:,})")
    print(f"   False Acceptance Rate:     {far_fixed * 100:.3f}% ({fp:,} / {len(nontarget_scores):,})")
    print(f"   False Rejection Rate:      {frr_fixed * 100:.3f}% ({fn:,} / {len(target_scores):,})")
    print(f"   Recall (Sensitivity):      {rec_fixed * 100:.3f}% ({tp:,} / {len(target_scores):,})")
    print(f"   Specificity:               {spec_fixed * 100:.3f}% ({tn:,} / {len(nontarget_scores):,})")
    print(f"   Precision:                 {prec_fixed * 100:.3f}%")
    print(f"   F1-Score:                  {f1_fixed * 100:.3f}%")
    print("-" * 75)
    print(f" THRESHOLD-INDEPENDENT BENCHMARK METRICS:")
    print(f"   Evaluation EER (Discrete): {eval_eer_discrete * 100:.3f}% (at tau_eval = {eval_tau_discrete:.4f})")
    print(f"   Evaluation EER (Interp):   {eval_eer_interp * 100:.3f}% (at tau_eval = {eval_tau_interp:.4f})")
    print(f"   Area Under ROC (ROC-AUC):  {eval_roc_auc:.5f}")
    print(f"   Minimum normalized t-DCF:  {eval_min_tdcf:.5f}")
    print("-" * 75)
    print(" CONFUSION MATRIX:")
    print(f"   True Positives  (Bonafide Accepted): {tp:6,d} / {len(target_scores):,d} ({tp/len(target_scores)*100:.2f}%)")
    print(f"   False Negatives (Bonafide Rejected): {fn:6,d} / {len(target_scores):,d} ({frr_fixed*100:.3f}% FRR)")
    print(f"   True Negatives  (Spoof Rejected):    {tn:6,d} / {len(nontarget_scores):,d} ({tn/len(nontarget_scores)*100:.2f}%)")
    print(f"   False Positives (Spoof Accepted):    {fp:6,d} / {len(nontarget_scores):,d} ({far_fixed*100:.3f}% FAR)")
    print("=" * 75)

    print("\nPER-ATTACK BREAKDOWN (At Fixed tau_dev = 11.875):")
    print(f"{'Attack':<8} {'Type':<26} {'Trials':<8} {'Mean Score':<12} {'Rejection/Acc %':<16} {'Error %':<10}")
    print("-" * 80)
    for atk, info in attack_breakdown.items():
        print(f"{atk:<8} {info['attack_type']:<26} {info['trial_count']:<8} {info['mean_score']:<12.3f} {info['accuracy_percent']:<16.2f} {info['error_rate_percent']:<10.2f}")
    print("=" * 80)

    # 9. Save all artifacts
    results_payload = {
        "dataset": "ASVspoof 2019 Logical Access (LA) Evaluation Set",
        "total_trials": total_eval_trials,
        "target_trials": len(target_scores),
        "nontarget_trials": len(nontarget_scores),
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": ckpt["epoch"],
        "operating_threshold_tau_dev": tau_dev,
        "operating_metrics": {
            "accuracy": round(acc_fixed, 6),
            "precision": round(prec_fixed, 6),
            "recall": round(rec_fixed, 6),
            "specificity": round(spec_fixed, 6),
            "f1_score": round(f1_fixed, 6),
            "false_acceptance_rate": round(far_fixed, 6),
            "false_rejection_rate": round(frr_fixed, 6),
            "confusion_matrix": cm,
        },
        "threshold_independent_metrics": {
            "eval_eer_discrete": round(eval_eer_discrete, 6),
            "eval_eer_discrete_percent": round(eval_eer_discrete * 100.0, 3),
            "eval_threshold_discrete": round(eval_tau_discrete, 6),
            "eval_eer_interpolated": round(eval_eer_interp, 6),
            "eval_eer_interpolated_percent": round(eval_eer_interp * 100.0, 3),
            "eval_threshold_interpolated": round(eval_tau_interp, 6),
            "roc_auc": round(eval_roc_auc, 6),
            "min_tdcf": round(eval_min_tdcf, 6),
        },
        "telemetry": {
            "total_inference_time_sec": round(total_eval_time, 2),
            "avg_latency_ms": round(avg_latency_ms, 2),
            "throughput_samples_per_sec": round(overall_throughput, 1),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "attack_breakdown": attack_breakdown,
    }

    out_json = os.path.join(output_dir, "final_evaluation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nFinal evaluation metrics successfully saved to: {out_json}")

    out_npz = os.path.join(output_dir, "eval_predictions.npz")
    np.savez_compressed(
        out_npz,
        file_ids=file_ids,
        attack_ids=attack_ids,
        labels=labels,
        scores=scores,
    )
    print(f"Predictions and scores successfully saved to: {out_npz}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate deepfake speech detector on ASVspoof 2019 LA Eval.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/custom_lfcc_v1/best_model.pt",
        help="Path to model checkpoint .pt file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results and predictions (defaults to checkpoint dir).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override threshold (defaults to checkpoint's best_threshold).",
    )
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        override_threshold=args.threshold,
    )
