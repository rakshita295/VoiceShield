"""
Unit tests for Custom LFCC-CNN-BiLSTM-Attention Deepfake Detector.
"""

import pytest
import torch
import numpy as np

from src.features.lfcc import LFCCFeatureExtractor
from src.models.lfcc_cnn_bilstm import LFCC_CNN_BiLSTM, TemporalAttentionPooling
from src.datasets.asvspoof2019 import (
    ASVSpoof2019LADataset,
    collate_variable_audio,
)


def test_lfcc_feature_extraction():
    """Verifies that Torchaudio LFCC extracts 60 cepstral dimensions and correct temporal frames."""
    extractor = LFCCFeatureExtractor(
        sample_rate=16000,
        n_filter=20,
        n_lfcc=20,
        n_fft=512,
        win_length=400,
        hop_length=160,
        with_deltas=True,
    )

    batch_size = 2
    samples = 64000  # 4 seconds
    waveform = torch.randn(batch_size, samples)

    features = extractor(waveform)
    # Expected: [B, 1, 60, T_frames] where T_frames = 64000 // 160 + 1 = 401
    assert features.shape == (batch_size, 1, 60, 401), f"Unexpected shape: {features.shape}"

    # Test 1D waveform input handling
    features_1d = extractor(torch.randn(samples))
    assert features_1d.shape == (1, 1, 60, 401)

    # Test valid CNN frame calculation
    lengths = torch.tensor([64000, 32000, 16000])
    valid_frames = extractor.compute_cnn_valid_frames(lengths, num_downsample_stages=4)
    assert valid_frames.shape == (3,)
    assert valid_frames[0].item() == 25  # 401 -> 200 -> 100 -> 50 -> 25
    assert valid_frames[1].item() == 12  # 201 -> 100 -> 50 -> 25 -> 12
    assert valid_frames[2].item() == 6   # 101 -> 50 -> 25 -> 12 -> 6


def test_model_architecture_and_parameters():
    """Verifies model structure, parameter counts, and 100% trainable status."""
    model = LFCC_CNN_BiLSTM()
    param_info = model.count_parameters()

    total_params = param_info["total_parameters"]
    trainable_params = param_info["trainable_parameters"]

    assert total_params == trainable_params, "All model parameters must be trainable."
    assert 1_500_000 <= total_params <= 2_500_000, (
        f"Expected parameter count in range [1.5M, 2.5M], got {total_params:,}"
    )

    # Verify all weight matrices have non-zero initialized weights
    for name, p in model.named_parameters():
        assert p.requires_grad, f"Parameter {name} does not require grad."
        if "weight" in name:
            assert not torch.all(p == 0), f"Weight {name} is all zeros."


def test_model_forward_backward_pass():
    """Verifies complete forward and backward propagation with gradient flow."""
    model = LFCC_CNN_BiLSTM()
    batch_size = 4
    audio = torch.randn(batch_size, 48000)
    lengths = torch.tensor([48000, 36000, 24000, 16000])

    logits, attn = model(audio, audio_lengths=lengths, return_attn=True)

    assert logits.shape == (batch_size, 2), f"Logits shape mismatch: {logits.shape}"
    assert attn.shape[0] == batch_size
    assert attn.ndim == 2

    loss = logits.sum()
    loss.backward()

    # Ensure gradients propagate to CNN, LSTM, Attention, and Classifier
    assert model.block1[0].weight.grad is not None
    assert model.block4[0].weight.grad is not None
    assert model.lstm.weight_ih_l0.grad is not None
    assert model.attention.proj.weight.grad is not None
    assert model.classifier[-1].weight.grad is not None


def test_temporal_attention_masking():
    """Verifies that attention pooling masks out padding frames."""
    attn_module = TemporalAttentionPooling(in_dim=256, attn_dim=64)
    batch_size = 2
    time_steps = 10
    h = torch.randn(batch_size, time_steps, 256)
    valid_lengths = torch.tensor([10, 4])  # First sample full, second has 4 valid frames

    context, alpha = attn_module(h, valid_lengths=valid_lengths)

    assert context.shape == (batch_size, 256)
    assert alpha.shape == (batch_size, time_steps)

    # Check softmax sum is 1.0 for each utterance
    sums = torch.sum(alpha, dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    # For the second item, frames >= 4 should have near-zero attention weight
    masked_weights = alpha[1, 4:]
    assert torch.all(masked_weights < 1e-6), f"Masked frames must have ~0 attention: {masked_weights}"


def test_differential_score_computation():
    """Verifies that compute_score computes bonafide - spoof."""
    model = LFCC_CNN_BiLSTM()
    model.eval()
    audio = torch.randn(2, 32000)

    with torch.no_grad():
        logits, _ = model(audio)
        expected_diff = logits[:, 1] - logits[:, 0]
        computed_diff = model.compute_score(audio)

    assert torch.allclose(expected_diff, computed_diff, atol=1e-5)


def test_variable_length_collate():
    """Verifies collate_variable_audio pads correctly and tracks true lengths."""
    wav1 = np.ones(1000, dtype=np.float32)
    wav2 = np.ones(500, dtype=np.float32)

    batch = [
        (wav1, 1, 1000, "file_001", "-"),
        (wav2, 0, 500, "file_002", "A01"),
    ]

    padded, labels, lengths, meta = collate_variable_audio(batch)

    assert padded.shape == (2, 1000)
    assert lengths.tolist() == [1000, 500]
    assert labels.tolist() == [1, 0]
    # Check that padded region of wav2 is strictly zeros
    assert torch.all(padded[1, 500:] == 0.0)
    assert torch.all(padded[1, :500] == 1.0)
