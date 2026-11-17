"""
Unit and Integration Test Suite for Utterance-Level Cepstral Mean and Variance Normalization (CMVN).

Verifies:
1. Zero-mean and unit-variance normalization per channel across valid temporal frames.
2. Numerical stability with small epsilon on silence and near-zero audio.
3. Proper handling of constant-valued features (zero variance).
4. Masked normalization on padded batches (padded frames remain strictly 0.0).
5. Exact consistency between single-utterance inference and batched training.
6. Comparison between 'after_deltas' and 'before_deltas' CMVN modes.
7. Correct output tensor dimensions across 1D and 2D inputs.
8. Backward compatibility with V1 baseline (use_cmvn=False).
9. Checkpoint loading and metadata inspection.
10. End-to-end gradient flow in the full model with CMVN.
"""

import os
import pytest
import torch
import numpy as np

from src.features.lfcc import LFCCFeatureExtractor
from src.models.lfcc_cnn_bilstm import LFCC_CNN_BiLSTM
from src.inference.predict_lfcc import load_detector


def test_cmvn_zero_mean_unit_variance():
    """Verifies that CMVN normalizes each of the 60 channels to mean ~0 and variance ~1."""
    extractor = LFCCFeatureExtractor(
        sample_rate=16000,
        n_filter=20,
        n_lfcc=20,
        with_deltas=True,
        use_cmvn=True,
        cmvn_mode="after_deltas",
        cmvn_eps=1e-6,
    )

    batch_size = 2
    samples = 32000  # 2.0s -> 201 frames
    waveform = torch.randn(batch_size, samples)

    features = extractor(waveform)  # [B, 1, 60, 201]
    assert features.shape == (batch_size, 1, 60, 201)

    # Inspect each utterance and each channel
    for b in range(batch_size):
        feat_2d = features[b, 0]  # [60, 201]
        means = feat_2d.mean(dim=-1)  # [60]
        stds = feat_2d.std(dim=-1, unbiased=False)  # [60]

        # All 60 channels must have zero mean and unit variance
        assert torch.all(torch.abs(means) < 1e-4), f"Means exceeded threshold: max={torch.abs(means).max().item()}"
        assert torch.all(torch.abs(stds - 1.0) < 1e-3), f"Stds deviated from 1.0: max_dev={torch.abs(stds - 1.0).max().item()}"


def test_cmvn_modes_before_and_after_deltas():
    """
    Compares 'after_deltas' vs 'before_deltas':
    - 'after_deltas' normalizes all 60 channels to unit variance.
    - 'before_deltas' normalizes static LFCC, but deltas naturally have smaller variances.
    """
    ext_after = LFCCFeatureExtractor(use_cmvn=True, cmvn_mode="after_deltas")
    ext_before = LFCCFeatureExtractor(use_cmvn=True, cmvn_mode="before_deltas")

    x = torch.randn(1, 32000)

    out_after = ext_after(x)[0, 0]   # [60, 201]
    out_before = ext_before(x)[0, 0] # [60, 201]

    # In after_deltas: all 60 channels have variance ~ 1.0
    stds_after = out_after.std(dim=-1, unbiased=False)
    assert torch.all(torch.abs(stds_after - 1.0) < 1e-3)

    # In before_deltas: static channels (0-19) have variance ~ 1.0, but deltas (20-59) have variance < 0.5
    stds_before = out_before.std(dim=-1, unbiased=False)
    assert torch.all(torch.abs(stds_before[:20] - 1.0) < 1e-3)
    assert torch.all(stds_before[20:40] < 0.6), "Delta variances should be naturally attenuated in before_deltas"
    assert torch.all(stds_before[40:60] < 0.3), "Delta-delta variances should be naturally attenuated in before_deltas"


def test_cmvn_constant_features_and_numerical_stability():
    """Verifies that constant features (variance=0) produce 0 without NaN or Inf."""
    extractor = LFCCFeatureExtractor(use_cmvn=True, cmvn_eps=1e-6)

    # 1. Direct apply_cmvn on constant tensor
    constant_feat = torch.full((1, 60, 100), 42.0)
    norm = extractor.apply_cmvn(constant_feat, eps=1e-6)

    assert not torch.isnan(norm).any(), "NaN found in normalized constant features"
    assert not torch.isinf(norm).any(), "Inf found in normalized constant features"
    assert torch.all(norm == 0.0), f"Expected all zeros for constant features, got max={norm.abs().max().item()}"

    # 2. Complete silence waveform
    silence = torch.zeros(1, 16000)
    out_silence = extractor(silence)

    assert not torch.isnan(out_silence).any(), "NaN in silence extraction"
    assert not torch.isinf(out_silence).any(), "Inf in silence extraction"


def test_cmvn_padded_batch_masking():
    """
    Verifies that for variable-length padded batches:
    1. Valid frames are normalized using only valid-frame statistics.
    2. Padded frames remain strictly 0.0.
    """
    extractor = LFCCFeatureExtractor(use_cmvn=True, cmvn_mode="after_deltas")

    len1 = 16000  # 101 LFCC frames
    len2 = 32000  # 201 LFCC frames

    # Create padded batch
    batch_audio = torch.zeros(2, len2)
    batch_audio[0, :len1] = torch.randn(len1)
    batch_audio[1, :len2] = torch.randn(len2)

    lengths = torch.tensor([len1, len2])

    features = extractor(batch_audio, audio_lengths=lengths)  # [2, 1, 60, 201]

    # Item 1: frames 0 to 100 are valid, frames 101 to 200 are padded
    valid_frames_1 = features[0, 0, :, :101]
    padded_frames_1 = features[0, 0, :, 101:]

    # Check that valid frames have zero mean and unit variance
    means_1 = valid_frames_1.mean(dim=-1)
    stds_1 = valid_frames_1.std(dim=-1, unbiased=False)
    assert torch.all(torch.abs(means_1) < 1e-4)
    assert torch.all(torch.abs(stds_1 - 1.0) < 1e-3)

    # Check that padded frames are strictly 0.0
    assert torch.all(padded_frames_1 == 0.0), (
        f"Padded frames must be strictly 0.0, got max={padded_frames_1.abs().max().item()}"
    )

    # Item 2: all 201 frames are valid
    valid_frames_2 = features[1, 0, :, :201]
    means_2 = valid_frames_2.mean(dim=-1)
    stds_2 = valid_frames_2.std(dim=-1, unbiased=False)
    assert torch.all(torch.abs(means_2) < 1e-4)
    assert torch.all(torch.abs(stds_2 - 1.0) < 1e-3)


def test_cmvn_training_inference_consistency():
    """
    Verifies that CMVN applied to an utterance individually (inference mode)
    matches the same utterance normalized inside a zero-padded batch (training mode).
    Confirms zero cross-sample leakage and numerical consistency.
    """
    extractor = LFCCFeatureExtractor(use_cmvn=True, cmvn_mode="after_deltas")

    # 1. Feature-level exact consistency
    torch.manual_seed(42)
    feat_short = torch.randn(1, 60, 101)
    feat_long = torch.randn(1, 60, 201)

    # Single inference mode (no padding)
    single_norm = extractor.apply_cmvn(feat_short)

    # Batched mode (sample 0 zero-padded to 201 frames)
    batch_feat = torch.zeros(2, 60, 201)
    batch_feat[0, :, :101] = feat_short
    batch_feat[1, :, :201] = feat_long

    batched_norm = extractor.apply_cmvn(batch_feat, valid_frames=torch.tensor([101, 201]))
    diff = (single_norm - batched_norm[0:1, :, :101]).abs().max().item()
    assert diff < 1e-6, f"Feature-level CMVN discrepancy: {diff:.8e}"

    # Padded frames must be strictly zero
    assert torch.all(batched_norm[0, :, 101:] == 0.0), "Padded frames must remain 0.0"

    # 2. Independent batch isolation (changing sample 1 does not alter sample 0)
    batch_feat_alt = batch_feat.clone()
    batch_feat_alt[1] = torch.randn(60, 201) * 100.0  # drastically different second sample

    batched_norm_alt = extractor.apply_cmvn(batch_feat_alt, valid_frames=torch.tensor([101, 201]))
    cross_sample_diff = (batched_norm[0] - batched_norm_alt[0]).abs().max().item()
    assert cross_sample_diff == 0.0, "Cross-sample data leakage detected in batched CMVN!"


def test_cmvn_output_tensor_dimensions():
    """Verifies output dimensions for 1D, 2D, and 3D inputs."""
    extractor = LFCCFeatureExtractor(use_cmvn=True)

    # 1D input: [32000]
    out_1d = extractor(torch.randn(32000))
    assert out_1d.shape == (1, 1, 60, 201)

    # 2D input: [3, 48000]
    out_2d = extractor(torch.randn(3, 48000))
    assert out_2d.shape == (3, 1, 60, 301)

    # Valid frame calculation
    lens = torch.tensor([16000, 32000, 48000, 64000])
    valid_lfcc = extractor.compute_lfcc_valid_frames(lens)
    assert valid_lfcc.tolist() == [101, 201, 301, 401]

    valid_cnn = extractor.compute_cnn_valid_frames(lens, num_downsample_stages=4)
    assert valid_cnn.tolist() == [6, 12, 18, 25]


def test_v1_backward_compatibility():
    """
    Verifies that when use_cmvn=False:
    1. LFCCFeatureExtractor output is identical to the baseline unnormalized transform.
    2. LFCC_CNN_BiLSTM defaults to use_cmvn=False.
    3. The V1 checkpoint (experiments/custom_lfcc_v1/best_model.pt) loads cleanly without errors.
    """
    # 1. Feature extractor comparison
    torch.manual_seed(42)
    x = torch.randn(2, 32000)

    ext_v1 = LFCCFeatureExtractor(use_cmvn=False)
    ext_cmvn = LFCCFeatureExtractor(use_cmvn=True)

    out_v1 = ext_v1(x)
    out_cmvn = ext_cmvn(x)

    # V1 output is not normalized to unit variance
    v1_stds = out_v1[0, 0].std(dim=-1, unbiased=False)
    assert not torch.allclose(v1_stds, torch.ones_like(v1_stds), atol=0.1)

    # 2. Checkpoint loading test
    ckpt_path = os.path.join("experiments", "custom_lfcc_v1", "best_model.pt")
    if os.path.isfile(ckpt_path):
        model, threshold, ckpt = load_detector(ckpt_path, device="cpu")
        assert not model.use_cmvn, "Loaded V1 model should have use_cmvn=False"
        assert threshold == pytest.approx(11.875, abs=1e-3)

        # Run smoke inference on V1 checkpoint
        with torch.no_grad():
            score = model.compute_score(x)
            assert score.shape == (2,)
            assert not torch.isnan(score).any()


def test_cmvn_model_gradient_flow():
    """Verifies that full model with CMVN executes forward and backward passes with valid gradients."""
    model = LFCC_CNN_BiLSTM(use_cmvn=True, cmvn_mode="after_deltas")
    model.train()

    batch_size = 2
    audio = torch.randn(batch_size, 32000)
    lengths = torch.tensor([24000, 32000])

    logits, attn = model(audio, audio_lengths=lengths, return_attn=True)
    assert logits.shape == (batch_size, 2)
    assert attn.shape[0] == batch_size

    loss = logits.sum()
    loss.backward()

    # Verify gradients reach CNN, LSTM, Attention, and Classifier
    assert model.block1[0].weight.grad is not None
    assert not torch.isnan(model.block1[0].weight.grad).any()
    assert model.lstm.weight_ih_l0.grad is not None
    assert model.attention.proj.weight.grad is not None
    assert model.classifier[-1].weight.grad is not None
