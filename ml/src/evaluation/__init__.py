"""Evaluation metrics and benchmarking modules for VoiceShield."""

from src.evaluation.metrics import (
    compute_det_curve,
    compute_eer,
    compute_roc_auc,
    compute_min_tdcf,
    evaluate_at_threshold,
)

__all__ = [
    "compute_det_curve",
    "compute_eer",
    "compute_roc_auc",
    "compute_min_tdcf",
    "evaluate_at_threshold",
]
