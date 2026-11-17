"""
Evaluation Metrics for VoiceShield Anti-Spoofing & Deepfake Detection.

Implements standard biometrics and ASVspoof challenge evaluation metrics:
- Equal Error Rate (EER) and operating threshold tau
- Detection Error Tradeoff (DET) curve: False Rejection Rate (FRR) & False Acceptance Rate (FAR)
- Minimum tandem Detection Cost Function (min t-DCF)
- Receiver Operating Characteristic (ROC) & Area Under Curve (ROC-AUC)
- Binary classification metrics: Accuracy, Precision, Recall, Specificity, F1-Score, Confusion Matrix
- Calibration routines separating Development calibration from Evaluation benchmarking.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np


def compute_det_curve(
    target_scores: np.ndarray,
    nontarget_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes False Rejection Rate (FRR), False Acceptance Rate (FAR), and thresholds.

    Convention: Higher score indicates greater likelihood of being TARGET (Bonafide).
    - Target trials classified as nontarget if score < threshold (False Rejection).
    - Nontarget trials classified as target if score >= threshold (False Acceptance).

    Args:
        target_scores: 1D array of scores for bonafide / genuine samples.
        nontarget_scores: 1D array of scores for spoof / synthetic samples.

    Returns:
        Tuple of (frr, far, thresholds).
    """
    target_scores = np.asarray(target_scores, dtype=np.float64)
    nontarget_scores = np.asarray(nontarget_scores, dtype=np.float64)

    n_target = target_scores.size
    n_nontarget = nontarget_scores.size

    if n_target == 0 or n_nontarget == 0:
        raise ValueError("Target and nontarget score arrays must both be non-empty.")

    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(n_target, dtype=np.int32), np.zeros(n_nontarget, dtype=np.int32)))

    # Sort in ascending order
    indices = np.argsort(all_scores, kind="mergesort")
    all_scores = all_scores[indices]
    labels = labels[indices]

    # Cumulative counts of targets and nontargets below each threshold
    tar_trial_sums = np.cumsum(labels)
    nontar_trial_sums = n_nontarget - (np.arange(1, len(all_scores) + 1) - tar_trial_sums)

    frr = np.concatenate(([0.0], tar_trial_sums / n_target))
    far = np.concatenate(([1.0], nontar_trial_sums / n_nontarget))
    thresholds = np.concatenate(([all_scores[0] - 1e-4], all_scores))

    return frr, far, thresholds


def compute_eer(
    target_scores: np.ndarray,
    nontarget_scores: np.ndarray
) -> Tuple[float, float]:
    """
    Computes Equal Error Rate (EER) and the corresponding operating decision threshold tau.

    Args:
        target_scores: 1D array of scores for bonafide / genuine samples.
        nontarget_scores: 1D array of scores for spoof / synthetic samples.

    Returns:
        Tuple of (eer, threshold) where eer is a float in [0.0, 1.0].
    """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_idx = np.argmin(abs_diffs)
    eer = float(np.mean((frr[min_idx], far[min_idx])))
    threshold = float(thresholds[min_idx])
    return eer, threshold


def compute_roc_auc(
    target_scores: np.ndarray,
    nontarget_scores: np.ndarray
) -> float:
    """
    Computes exact Area Under the ROC Curve (ROC-AUC) using Mann-Whitney U statistic.
    Target (bonafide) is treated as positive class (1), nontarget (spoof) as negative (0).
    Higher score indicates greater likelihood of being positive (bonafide).

    Returns:
        Exact ROC-AUC as a float in [0.0, 1.0].
    """
    target_scores = np.asarray(target_scores, dtype=np.float64)
    nontarget_scores = np.asarray(nontarget_scores, dtype=np.float64)

    n_pos = len(target_scores)
    n_neg = len(nontarget_scores)

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Target and nontarget score arrays must both be non-empty.")

    # Combine scores and rank them
    all_scores = np.concatenate((target_scores, nontarget_scores))
    ranks = np.empty(len(all_scores), dtype=np.float64)
    sorter = np.argsort(all_scores)
    inv_sorter = np.empty_like(sorter)
    inv_sorter[sorter] = np.arange(len(all_scores))

    # Fast average rank calculation for ties
    sorted_scores = all_scores[sorter]
    unique_mask = np.empty(len(all_scores), dtype=bool)
    unique_mask[0] = True
    unique_mask[1:] = sorted_scores[1:] != sorted_scores[:-1]
    unique_indices = np.flatnonzero(unique_mask)

    repeat_counts = np.diff(np.append(unique_indices, len(all_scores)))
    avg_ranks = unique_indices + (repeat_counts - 1) / 2.0 + 1.0  # 1-based ranks
    ranks = np.repeat(avg_ranks, repeat_counts)[inv_sorter]

    rank_sum_pos = np.sum(ranks[:n_pos])
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    auc = u_stat / (n_pos * n_neg)

    return float(max(0.0, min(1.0, auc)))



def evaluate_at_threshold(
    target_scores: np.ndarray,
    nontarget_scores: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """
    Evaluates classification performance at a predefined fixed threshold.
    Scores >= threshold are predicted as Bonafide (Class 1).
    Scores < threshold are predicted as Spoof (Class 0).

    Args:
        target_scores: Array of scores for true bonafide samples.
        nontarget_scores: Array of scores for true spoof samples.
        threshold: Fixed decision threshold (calibrated from development data).

    Returns:
        Dictionary containing TP, FP, TN, FN, Accuracy, Precision, Recall, Specificity, F1.
    """
    target_scores = np.asarray(target_scores, dtype=np.float64)
    nontarget_scores = np.asarray(nontarget_scores, dtype=np.float64)

    # Bonafide is positive (Class 1)
    tp = int(np.sum(target_scores >= threshold))
    fn = int(np.sum(target_scores < threshold))

    # Spoof is negative (Class 0)
    tn = int(np.sum(nontarget_scores < threshold))
    fp = int(np.sum(nontarget_scores >= threshold))

    total = tp + fn + tn + fp
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity / TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    frr = fn / len(target_scores) if len(target_scores) > 0 else 0.0
    far = fp / len(nontarget_scores) if len(nontarget_scores) > 0 else 0.0
    hter = (frr + far) / 2.0

    return {
        "threshold": float(threshold),
        "total_samples": total,
        "true_positives": tp,
        "false_negatives": fn,
        "true_negatives": tn,
        "false_positives": fp,
        "confusion_matrix": {
            "TP_bonafide_accepted": tp,
            "FN_bonafide_rejected": fn,
            "TN_spoof_rejected": tn,
            "FP_spoof_accepted": fp,
        },
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall_sensitivity": round(float(recall), 6),
        "specificity": round(float(specificity), 6),
        "f1_score": round(float(f1), 6),
        "false_rejection_rate": round(float(frr), 6),
        "false_acceptance_rate": round(float(far), 6),
        "half_total_error_rate": round(float(hter), 6),
    }


def compute_min_tdcf(
    target_scores: np.ndarray,
    nontarget_scores: np.ndarray,
    p_spoof: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 10.0
) -> Tuple[float, float]:
    """
    Computes minimum normalized tandem Detection Cost Function (min t-DCF)
    under standard ASVspoof parameters.

    Args:
        target_scores: Bonafide scores.
        nontarget_scores: Spoof scores.
        p_spoof: Prior probability of spoofing attack (default: 0.05).
        c_miss: Cost of falsely rejecting target (bonafide).
        c_fa: Cost of falsely accepting spoof.

    Returns:
        Tuple of (min_tdcf, threshold_at_min_tdcf).
    """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)

    c_det = c_miss * (1.0 - p_spoof) * frr + c_fa * p_spoof * far
    c_def = min(c_miss * (1.0 - p_spoof), c_fa * p_spoof)
    norm_tdcf = c_det / c_def

    min_idx = np.argmin(norm_tdcf)
    min_tdcf = float(norm_tdcf[min_idx])
    thresh = float(thresholds[min_idx])
    return min_tdcf, thresh
