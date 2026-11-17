"""
Audio Preprocessing Module for VoiceShield.

Provides standard audio conditioning for VoiceShield models:
- Mono conversion (averages multi-channel audio)
- Resampling to target rate (16,000 Hz)
- Duration clipping (up to 4.0 seconds = 64,000 samples)
- Sample length tracking for dynamic masking
"""

import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
import soundfile as sf
import scipy.signal
import torch

SAMPLE_RATE: int = 16000
DEFAULT_MAX_DURATION_SEC: float = 4.0
DEFAULT_MAX_SAMPLES: int = int(SAMPLE_RATE * DEFAULT_MAX_DURATION_SEC)  # 64,000 samples


def preprocess_waveform(
    wav: np.ndarray,
    sr: int,
    target_sr: int = SAMPLE_RATE,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Conditions raw audio waveform for VoiceShield inference.

    Args:
        wav: 1D or 2D numpy array of audio samples (float32 in [-1.0, 1.0]).
        sr: Original sampling rate in Hz.
        target_sr: Target sampling rate (default: 16000).
        max_samples: Maximum sample count to retain (default: 64000).

    Returns:
        Tuple of:
        - waveform_tensor: torch.FloatTensor of shape [1, T_samples]
        - length_tensor: torch.LongTensor of shape [1] containing true sample count
        - metadata: Dictionary containing original and processed attributes
    """
    orig_sr = int(sr)
    orig_channels = 1 if wav.ndim == 1 else int(wav.shape[1])
    orig_duration_sec = float(len(wav) / sr)

    # 1. Multi-channel to mono conversion
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    # 2. Resample to target rate if required
    if sr != target_sr:
        new_len = int(round(len(wav) * target_sr / sr))
        wav = scipy.signal.resample(wav, new_len).astype(np.float32)

    total_samples = len(wav)

    # 3. Truncate if exceeding max_samples
    if total_samples > max_samples:
        wav = wav[:max_samples]
        true_len = max_samples
        action = "truncated_to_max"
    else:
        true_len = total_samples
        action = "natural_length"

    tensor = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)  # [1, T]
    length_tensor = torch.tensor([true_len], dtype=torch.long)

    metadata = {
        "original_sample_rate": orig_sr,
        "original_channels": orig_channels,
        "original_duration_sec": round(orig_duration_sec, 3),
        "target_sample_rate": target_sr,
        "processed_samples": true_len,
        "processed_duration_sec": round(true_len / target_sr, 3),
        "conditioning_action": action,
    }
    return tensor, length_tensor, metadata


def load_and_preprocess_audio(
    audio_path: str,
    target_sr: int = SAMPLE_RATE,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Loads an audio file from disk and applies VoiceShield preprocessing.

    Args:
        audio_path: Absolute or relative path to target audio file (.wav, .flac, etc.).
        target_sr: Target sample rate (16 kHz).
        max_samples: Maximum duration cutoff (64,000 samples).

    Returns:
        Tuple of (waveform_tensor [1, T], length_tensor [1], metadata_dict).
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    wav, sr = sf.read(audio_path, dtype="float32")
    tensor, length_tensor, meta = preprocess_waveform(
        wav, sr, target_sr=target_sr, max_samples=max_samples
    )
    meta["audio_path"] = audio_path
    return tensor, length_tensor, meta
