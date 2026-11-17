"""
Unit and Integration Tests for VoiceShield Evaluation Metrics.

Tests:
1. Synthetic DET curve, EER, and threshold calculations.
2. ROC-AUC calculation bounds and known values.
3. Confusion matrix, precision, recall, F1, and HTER calculations.
4. min t-DCF calculation sanity.
"""

import os
import pytest
import numpy as np

from src.evaluation.metrics import (
    compute_det_curve,
    compute_eer,
    compute_roc_auc,
    compute_min_tdcf,
    evaluate_at_threshold,
)


def test_synthetic_det_and_eer():
    """Verify DET and EER computation with known perfectly separated distributions."""
    # Synthetic scores: Bonafide scores strictly higher than spoof
    target_scores = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    nontarget_scores = np.array([-6.0, -5.0, -4.0, -3.0, -2.0])

    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    eer, threshold = compute_eer(target_scores, nontarget_scores)

    # In perfectly separated data, EER should be 0.0
    assert eer == 0.0, f"Expected 0.0 EER for separated data, got {eer}"
    assert -2.0 <= threshold <= 2.0, f"Threshold should be between sets, got {threshold}"

    # Synthetic overlapping scores: 50% overlap
    target_overlap = np.array([0.0, 1.0, 2.0])
    nontarget_overlap = np.array([0.0, 1.0, 2.0])
    eer_overlap, _ = compute_eer(target_overlap, nontarget_overlap)
    assert 0.0 <= eer_overlap <= 1.0


def test_roc_auc_bounds():
    """Verify ROC-AUC calculation bounds [0.0, 1.0]."""
    # Perfect classifier
    bona_perf = np.array([10.0, 12.0, 15.0])
    spoof_perf = np.array([-10.0, -12.0, -15.0])
    auc_perf = compute_roc_auc(bona_perf, spoof_perf)
    assert auc_perf == 1.0

    # Inverted classifier
    auc_inv = compute_roc_auc(spoof_perf, bona_perf)
    assert auc_inv == 0.0

    # Random / overlapping
    bona_rand = np.array([1.0, 2.0, 3.0])
    spoof_rand = np.array([1.0, 2.0, 3.0])
    auc_rand = compute_roc_auc(bona_rand, spoof_rand)
    assert 0.0 <= auc_rand <= 1.0


def test_evaluate_at_threshold_metrics():
    """Verify accuracy, FAR, FRR, precision, recall, and F1 calculations at fixed threshold."""
    # 4 targets, 4 nontargets
    # target: [1.0, 2.0, 3.0, 4.0] (all > 0)
    # nontarget: [-3.0, -2.0, -1.0, 0.5] (one FP at 0.5 > 0)
    bona = np.array([1.0, 2.0, 3.0, 4.0])
    spoof = np.array([-3.0, -2.0, -1.0, 0.5])

    threshold = 0.0
    res = evaluate_at_threshold(bona, spoof, threshold=threshold)

    # TP = 4, FN = 0, TN = 3, FP = 1
    assert res["true_positives"] == 4
    assert res["false_negatives"] == 0
    assert res["true_negatives"] == 3
    assert res["false_positives"] == 1
    assert res["accuracy"] == 7.0 / 8.0      # 7/8 = 0.875
    assert res["false_rejection_rate"] == 0.0 # 0/4
    assert res["false_acceptance_rate"] == 0.25 # 1/4
    assert res["recall_sensitivity"] == 1.0  # 4/4
    assert res["specificity"] == 0.75        # 3/4
    assert res["precision"] == 4.0 / 5.0     # 4/(4+1) = 0.8
    expected_f1 = (2 * 0.8 * 1.0) / (0.8 + 1.0)
    assert abs(res["f1_score"] - expected_f1) < 1e-4


def test_min_tdcf_computation():
    """Verify min t-DCF returns a non-negative float."""
    bona = np.array([5.0, 6.0, 7.0, 8.0])
    spoof = np.array([-5.0, -6.0, -7.0, -8.0])
    min_tdcf, thresh = compute_min_tdcf(bona, spoof)
    assert 0.0 <= min_tdcf <= 1.0
