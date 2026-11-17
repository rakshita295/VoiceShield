"""
Inference CLI for Custom LFCC-CNN-BiLSTM-Attention Deepfake Detector.

Loads trained VoiceShield-LFCC model checkpoint and evaluates target audio files.

Usage:
    python src/inference/predict_lfcc.py datasets/samples/genuine_human.wav datasets/samples/synthetic_ai.wav
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import soundfile as sf
import scipy.signal
import torch

from src.models.lfcc_cnn_bilstm import LFCC_CNN_BiLSTM

TARGET_SAMPLE_RATE = 16000
MAX_SAMPLES = 64000  # 4.0 seconds @ 16 kHz


def preprocess_audio(audio_path: str) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Decodes and conditions audio for LFCC model:
    - Mono conversion
    - Resampling to 16,000 Hz if necessary
    - Center/start slice if length > 64,000 samples
    - Length tracking

    Returns:
        Tuple of (waveform_tensor [1, T], length_tensor [1], metadata_dict).
    """
    wav, sr = sf.read(audio_path, dtype="float32")
    orig_sr = int(sr)
    orig_duration = len(wav) / sr

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if sr != TARGET_SAMPLE_RATE:
        new_len = int(round(len(wav) * TARGET_SAMPLE_RATE / sr))
        wav = scipy.signal.resample(wav, new_len).astype(np.float32)

    total_samples = len(wav)
    if total_samples > MAX_SAMPLES:
        wav = wav[:MAX_SAMPLES]
        true_len = MAX_SAMPLES
        action = "truncated_to_4s"
    else:
        true_len = total_samples
        action = "natural_length"

    tensor = torch.from_numpy(wav).unsqueeze(0)  # [1, T]
    length_tensor = torch.tensor([true_len], dtype=torch.long)

    metadata = {
        "audio_path": audio_path,
        "original_sample_rate": orig_sr,
        "original_duration_sec": round(orig_duration, 3),
        "processed_samples": true_len,
        "action": action,
    }
    return tensor, length_tensor, metadata


def load_detector(
    checkpoint_path: str = "experiments/custom_lfcc_v1/best_model.pt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[LFCC_CNN_BiLSTM, float, Dict[str, Any]]:
    """
    Loads LFCC detector from checkpoint.

    Returns:
        Tuple of (model, calibrated_threshold_tau, checkpoint_dict).
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    target_device = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=target_device)

    # Detect CMVN configuration from checkpoint metadata (backward-compatible with V1)
    config = ckpt.get("config", {})
    use_cmvn = config.get("use_cmvn", False)
    cmvn_mode = config.get("cmvn_mode", "after_deltas")
    cmvn_eps = float(config.get("cmvn_eps", 1e-6))

    model = LFCC_CNN_BiLSTM(
        use_cmvn=use_cmvn,
        cmvn_mode=cmvn_mode,
        cmvn_eps=cmvn_eps,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(target_device)
    model.eval()

    threshold = float(ckpt.get("best_threshold", 0.0))
    return model, threshold, ckpt


def predict_file(
    model: LFCC_CNN_BiLSTM,
    audio_path: str,
    threshold: float,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    """Runs end-to-end inference on a single audio file."""
    t0 = time.time()
    tensor, length_tensor, meta = preprocess_audio(audio_path)
    t_preproc = (time.time() - t0) * 1000.0

    target_device = torch.device(device)
    tensor = tensor.to(target_device)
    length_tensor = length_tensor.to(target_device)

    t1 = time.time()
    with torch.no_grad():
        logits, _ = model(tensor, audio_lengths=length_tensor)
        # Logit 0: Spoof, Logit 1: Bonafide
        spoof_logit = float(logits[0, 0].item())
        bonafide_logit = float(logits[0, 1].item())
        diff_score = bonafide_logit - spoof_logit

        probs = torch.softmax(logits, dim=-1)
        prob_spoof = float(probs[0, 0].item())
        prob_bonafide = float(probs[0, 1].item())

    t_infer = (time.time() - t1) * 1000.0
    t_total = t_preproc + t_infer

    is_bonafide = (diff_score >= threshold)
    label_str = "GENUINE_HUMAN" if is_bonafide else "AI-GENERATED / SYNTHETIC"
    confidence = prob_bonafide if is_bonafide else prob_spoof

    return {
        "file": os.path.basename(audio_path),
        "path": audio_path,
        "prediction": label_str,
        "is_bonafide": is_bonafide,
        "confidence": round(confidence * 100.0, 2),
        "differential_score": round(diff_score, 5),
        "operating_threshold": round(threshold, 5),
        "spoof_logit": round(spoof_logit, 4),
        "bonafide_logit": round(bonafide_logit, 4),
        "prob_spoof": round(prob_spoof, 5),
        "prob_bonafide": round(prob_bonafide, 5),
        "latency_preproc_ms": round(t_preproc, 2),
        "latency_infer_ms": round(t_infer, 2),
        "latency_total_ms": round(t_total, 2),
        "metadata": meta,
    }


def main():
    parser = argparse.ArgumentParser(description="VoiceShield LFCC Deepfake Detector Inference")
    parser.add_argument("files", nargs="+", help="Audio file path(s) to classify.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/custom_lfcc_v1/best_model.pt",
        help="Path to trained checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device ('cuda' or 'cpu').",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" VoiceShield: Custom LFCC-CNN-BiLSTM Inference Engine")
    print("=" * 70)
    print(f" Device:     {args.device}")
    print(f" Checkpoint: {args.checkpoint}")

    model, threshold, ckpt = load_detector(args.checkpoint, args.device)
    print(f" Calibrated Threshold (tau): {threshold:.6f} (from dev partition)")
    print(f" Checkpoint Epoch:           {ckpt.get('epoch', 'N/A')}")
    print(f" Checkpoint Best Dev EER:    {ckpt.get('best_val_eer', 0.0) * 100.0:.3f}%")
    print("=" * 70)

    for file_path in args.files:
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            continue
        res = predict_file(model, file_path, threshold, args.device)
        print(f"\n--- File: {res['file']} ---")
        print(f"  Prediction:          >> {res['prediction']} << (Confidence: {res['confidence']}%)")
        print(f"  Differential Score:  {res['differential_score']} (Threshold: {res['operating_threshold']})")
        print(f"  Probabilities:       Spoof: {res['prob_spoof']:.4f} | Bonafide: {res['prob_bonafide']:.4f}")
        print(f"  Latency:             Preproc: {res['latency_preproc_ms']:.1f}ms | Infer: {res['latency_infer_ms']:.1f}ms | Total: {res['latency_total_ms']:.1f}ms")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
