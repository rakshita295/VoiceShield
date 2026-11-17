"""
Comprehensive Audit Script for Development Predictions, Loss Analysis, and Metrics.
"""

import os
import sys
import json
import time
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
    evaluate_at_threshold,
)


def run_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running audit on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    checkpoint_path = "experiments/custom_lfcc_v1/best_model.pt"
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"  Checkpoint Epoch: {ckpt['epoch']}")
    print(f"  Stored Best Dev EER: {ckpt['best_val_eer'] * 100:.3f}%")
    print(f"  Stored Operating Threshold: {ckpt['best_threshold']}")

    # Instantiate model and load state_dict
    model = LFCC_CNN_BiLSTM()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Create Dev DataLoader
    dev_loader = create_asvspoof_dataloader(
        split="dev",
        batch_size=32,
        shuffle=False,
    )
    total_trials = len(dev_loader.dataset)
    print(f"Total development trials to evaluate: {total_trials:,}")
    print(f"  Bonafide: {dev_loader.dataset.n_bonafide:,} | Spoof: {dev_loader.dataset.n_spoof:,}")

    # Loss criterion with exact weights
    class_weights = dev_loader.dataset.get_class_weights().to(device)
    # Using training class weights as used during training:
    # w_spoof = 25380 / (2 * 22800) = 0.5565789
    # w_bonafide = 25380 / (2 * 2580) = 4.9186046
    train_weights = torch.tensor([0.5565789, 4.9186046], device=device)
    criterion_weighted = nn.CrossEntropyLoss(weight=train_weights, reduction="none")
    criterion_unweighted = nn.CrossEntropyLoss(reduction="none")

    all_losses_weighted = []
    all_losses_unweighted = []
    all_spoof_logits = []
    all_bonafide_logits = []
    all_diff_scores = []
    all_labels = []
    all_file_ids = []
    all_attack_ids = []

    print("Evaluating full development split...")
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (audio, labels, lengths, meta) in enumerate(dev_loader, start=1):
            audio = audio.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits, _ = model(audio, audio_lengths=lengths)
                loss_w = criterion_weighted(logits, labels)
                loss_u = criterion_unweighted(logits, labels)

            s_logits = logits[:, 0].cpu().numpy()
            b_logits = logits[:, 1].cpu().numpy()
            diffs = (logits[:, 1] - logits[:, 0]).cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_losses_weighted.extend(loss_w.cpu().numpy().tolist())
            all_losses_unweighted.extend(loss_u.cpu().numpy().tolist())
            all_spoof_logits.extend(s_logits.tolist())
            all_bonafide_logits.extend(b_logits.tolist())
            all_diff_scores.extend(diffs.tolist())
            all_labels.extend(labels_np.tolist())
            all_file_ids.extend(meta["file_ids"])
            all_attack_ids.extend(meta["attack_ids"])

            if batch_idx % 150 == 0 or batch_idx == len(dev_loader):
                print(f"  Processed {len(all_labels):,}/{total_trials:,} trials...", flush=True)

    elapsed = time.time() - t0
    print(f"Evaluation completed in {elapsed:.2f} seconds ({total_trials / elapsed:.1f} trials/s).")

    # Convert to arrays
    all_losses_weighted = np.array(all_losses_weighted, dtype=np.float64)
    all_losses_unweighted = np.array(all_losses_unweighted, dtype=np.float64)
    all_spoof_logits = np.array(all_spoof_logits, dtype=np.float64)
    all_bonafide_logits = np.array(all_bonafide_logits, dtype=np.float64)
    all_diff_scores = np.array(all_diff_scores, dtype=np.float64)
    all_labels = np.array(all_labels, dtype=np.int32)

    # 1. Overall Loss Metrics
    mean_weighted_loss = float(np.mean(all_losses_weighted))
    mean_unweighted_loss = float(np.mean(all_losses_unweighted))
    median_weighted_loss = float(np.median(all_losses_weighted))

    bonafide_mask = (all_labels == 1)
    spoof_mask = (all_labels == 0)

    loss_bonafide_mean = float(np.mean(all_losses_weighted[bonafide_mask]))
    loss_spoof_mean = float(np.mean(all_losses_weighted[spoof_mask]))

    print("\n" + "=" * 70)
    print(" 1. LOSS ANALYSIS (Why did Dev Loss increase to 0.24976?)")
    print("=" * 70)
    print(f"  Mean Weighted Dev Loss:   {mean_weighted_loss:.5f} (Reported in training: 0.24976)")
    print(f"  Mean Unweighted Dev Loss: {mean_unweighted_loss:.5f}")
    print(f"  Median Weighted Dev Loss: {median_weighted_loss:.5f}")
    print(f"  Bonafide Weighted Loss:   {loss_bonafide_mean:.5f} (Weight = 4.9186)")
    print(f"  Spoof Weighted Loss:      {loss_spoof_mean:.5f} (Weight = 0.5566)")

    # Outlier analysis
    loss_95th = np.percentile(all_losses_weighted, 95)
    loss_99th = np.percentile(all_losses_weighted, 99)
    outliers_gt_1 = np.sum(all_losses_weighted > 1.0)
    outliers_gt_5 = np.sum(all_losses_weighted > 5.0)
    outliers_gt_10 = np.sum(all_losses_weighted > 10.0)
    print(f"  95th Percentile Loss:     {loss_95th:.5f}")
    print(f"  99th Percentile Loss:     {loss_99th:.5f}")
    print(f"  Samples with loss > 1.0:  {outliers_gt_1} ({outliers_gt_1 / total_trials * 100:.2f}%)")
    print(f"  Samples with loss > 5.0:  {outliers_gt_5} ({outliers_gt_5 / total_trials * 100:.2f}%)")
    print(f"  Samples with loss > 10.0: {outliers_gt_10} ({outliers_gt_10 / total_trials * 100:.2f}%)")

    # 2. Score and Logit Statistics
    target_scores = all_diff_scores[bonafide_mask]
    nontarget_scores = all_diff_scores[spoof_mask]

    print("\n" + "=" * 70)
    print(" 2. LOGIT & SCORE DISTRIBUTIONS")
    print("=" * 70)
    print(f"  Bonafide Logits (Target):    Mean = {np.mean(all_bonafide_logits[bonafide_mask]):.3f}, Std = {np.std(all_bonafide_logits[bonafide_mask]):.3f}")
    print(f"  Spoof Logits (Target):       Mean = {np.mean(all_spoof_logits[bonafide_mask]):.3f}, Std = {np.std(all_spoof_logits[bonafide_mask]):.3f}")
    print(f"  Diff Score (Target/Bona):    Mean = {np.mean(target_scores):.3f}, Std = {np.std(target_scores):.3f}, Min = {np.min(target_scores):.3f}, Max = {np.max(target_scores):.3f}")
    print(f"  Diff Score (Nontarget/Spoof):Mean = {np.mean(nontarget_scores):.3f}, Std = {np.std(nontarget_scores):.3f}, Min = {np.min(nontarget_scores):.3f}, Max = {np.max(nontarget_scores):.3f}")
    print(f"  Separation Distance (Means): {np.mean(target_scores) - np.mean(nontarget_scores):.3f} score units")

    # 3. Independent EER and Threshold Calculation
    recomputed_eer, recomputed_tau = compute_eer(target_scores, nontarget_scores)
    recomputed_auc = compute_roc_auc(target_scores, nontarget_scores)

    print("\n" + "=" * 70)
    print(" 3. INDEPENDENT EER & THRESHOLD RECALCULATION")
    print("=" * 70)
    print(f"  Recomputed EER:            {recomputed_eer * 100:.3f}% (Stored: 0.778%)")
    print(f"  Recomputed Threshold tau:  {recomputed_tau:.6f} (Stored: 11.875000)")
    print(f"  Recomputed ROC-AUC:        {recomputed_auc:.5f} (Stored: 0.99955)")

    # 4. Independent Confusion Matrix & Metrics at tau
    metrics = evaluate_at_threshold(target_scores, nontarget_scores, recomputed_tau)
    cm = metrics["confusion_matrix"]

    tp = cm["TP_bonafide_accepted"]
    fn = cm["FN_bonafide_rejected"]
    tn = cm["TN_spoof_rejected"]
    fp = cm["FP_spoof_accepted"]

    far = fp / (tn + fp)
    frr = fn / (tp + fn)
    acc = (tp + tn) / (tp + fn + tn + fp)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)

    print("\n" + "=" * 70)
    print(" 4. CONFUSION MATRIX & RATE VERIFICATION")
    print("=" * 70)
    print(f"  True Positives  (Bonafide Accepted): {tp:,} / {len(target_scores):,} ({tp / len(target_scores) * 100:.2f}%)")
    print(f"  False Negatives (Bonafide Rejected): {fn:,} / {len(target_scores):,} ({frr * 100:.3f}% FRR)")
    print(f"  True Negatives  (Spoof Rejected):    {tn:,} / {len(nontarget_scores):,} ({tn / len(nontarget_scores) * 100:.2f}%)")
    print(f"  False Positives (Spoof Accepted):    {fp:,} / {len(nontarget_scores):,} ({far * 100:.3f}% FAR)")
    print(f"  Overall Accuracy:                    {acc * 100:.3f}%")
    print(f"  Precision:                           {prec * 100:.2f}%")
    print(f"  Recall:                              {rec * 100:.2f}%")
    print(f"  F1 Score:                            {f1 * 100:.2f}%")

    # 5. Save audit results
    audit_data = {
        "checkpoint": checkpoint_path,
        "epoch": ckpt["epoch"],
        "total_trials": total_trials,
        "target_count": len(target_scores),
        "nontarget_count": len(nontarget_scores),
        "recomputed_eer": round(float(recomputed_eer), 6),
        "recomputed_eer_percent": round(float(recomputed_eer) * 100.0, 3),
        "recomputed_tau": round(float(recomputed_tau), 6),
        "recomputed_roc_auc": round(float(recomputed_auc), 6),
        "mean_weighted_loss": round(mean_weighted_loss, 5),
        "mean_unweighted_loss": round(mean_unweighted_loss, 5),
        "median_weighted_loss": round(median_weighted_loss, 5),
        "confusion_matrix": cm,
        "metrics_at_tau": {
            "accuracy": round(acc, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "specificity": round(tn / (tn + fp), 6),
            "f1_score": round(f1, 6),
            "far": round(far, 6),
            "frr": round(frr, 6),
        },
        "score_statistics": {
            "bonafide_mean": round(float(np.mean(target_scores)), 4),
            "bonafide_std": round(float(np.std(target_scores)), 4),
            "bonafide_min": round(float(np.min(target_scores)), 4),
            "bonafide_max": round(float(np.max(target_scores)), 4),
            "spoof_mean": round(float(np.mean(nontarget_scores)), 4),
            "spoof_std": round(float(np.std(nontarget_scores)), 4),
            "spoof_min": round(float(np.min(nontarget_scores)), 4),
            "spoof_max": round(float(np.max(nontarget_scores)), 4),
        }
    }

    out_json = "experiments/custom_lfcc_v1/dev_audit_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print(f"\nAudit results successfully saved to: {out_json}")


if __name__ == "__main__":
    run_audit()
