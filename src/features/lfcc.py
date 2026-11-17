"""
LFCC (Linear Frequency Cepstral Coefficients) Feature Extraction Module.

Extracts spectral representations from raw audio waveforms using Torchaudio:
- Linear filter banks (more sensitive to high-frequency synthetic artifacts than Mel scale)
- Computes static, delta, and delta-delta cepstral coefficients (60 total dimensions)
- Prepares 4D tensors [Batch, 1, Channels, Time] formatted for 2D-CNN architectures.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torchaudio


class LFCCFeatureExtractor(nn.Module):
    """
    Torchaudio-native LFCC feature extractor with optional utterance-level CMVN.

    Args:
        sample_rate: Audio sampling rate in Hz (default: 16000).
        n_filter: Number of linear filters in the filterbank (default: 20).
        n_lfcc: Number of LFCC cepstral coefficients to retain (default: 20).
        n_fft: FFT window length (default: 512).
        win_length: Window analysis length in samples (default: 400, i.e., 25ms @ 16kHz).
        hop_length: Frame hop step in samples (default: 160, i.e., 10ms @ 16kHz).
        with_deltas: If True, computes delta and delta-delta coefficients (3 * n_lfcc = 60).
        log_lf: If True, uses log of filter bank energies (default: True).
        use_cmvn: If True, applies utterance-level Cepstral Mean and Variance Normalization.
        cmvn_mode: 'after_deltas' (normalizes static + delta + delta-delta to unit variance)
                   or 'before_deltas' (normalizes static LFCC before computing deltas).
        cmvn_eps: Small positive constant for numerical stability (default: 1e-6).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_filter: int = 20,
        n_lfcc: int = 20,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        with_deltas: bool = True,
        log_lf: bool = True,
        use_cmvn: bool = False,
        cmvn_mode: str = "after_deltas",
        cmvn_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_filter = n_filter
        self.n_lfcc = n_lfcc
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.with_deltas = with_deltas
        self.log_lf = log_lf
        self.use_cmvn = use_cmvn
        self.cmvn_mode = cmvn_mode.lower().strip()
        self.cmvn_eps = cmvn_eps

        if self.cmvn_mode not in {"after_deltas", "before_deltas"}:
            raise ValueError(f"Unknown cmvn_mode '{self.cmvn_mode}'. Choose 'after_deltas' or 'before_deltas'.")

        self.num_channels = (3 * n_lfcc) if with_deltas else n_lfcc

        # Torchaudio LFCC transform
        speckwargs = {
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "center": True,
            "pad_mode": "reflect",
        }
        self.lfcc_transform = torchaudio.transforms.LFCC(
            sample_rate=self.sample_rate,
            n_filter=self.n_filter,
            n_lfcc=self.n_lfcc,
            log_lf=self.log_lf,
            speckwargs=speckwargs,
        )

    @staticmethod
    def apply_cmvn(
        features: torch.Tensor,
        valid_frames: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Applies utterance-level Cepstral Mean and Variance Normalization (CMVN).

        Args:
            features: Tensor of shape [B, C, T] containing cepstral coefficients.
            valid_frames: Optional 1D tensor [B] indicating number of valid frames per utterance.
                          If provided, frames beyond valid_frames are excluded from mean/variance
                          computations and are strictly zero-masked in the output.
            eps: Small positive constant for numerical stability to prevent division by zero.

        Returns:
            Normalized tensor of shape [B, C, T] with zero mean and unit variance per channel
            across valid frames. Padded frames (if any) are set to 0.0.
        """
        B, C, T = features.shape

        if valid_frames is None:
            mean = features.mean(dim=-1, keepdim=True)
            var = features.var(dim=-1, keepdim=True, unbiased=False)
            std = torch.sqrt(var + eps)
            return (features - mean) / std

        device = features.device
        clamped_lengths = torch.clamp(valid_frames.to(device), min=1, max=T)
        frame_idx = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
        mask = (frame_idx < clamped_lengths.unsqueeze(1)).unsqueeze(1).to(dtype=features.dtype)  # [B, 1, T]

        n_valid = clamped_lengths.unsqueeze(1).unsqueeze(2).to(dtype=features.dtype)  # [B, 1, 1]

        # 1. Masked Mean across valid frames
        mean = (features * mask).sum(dim=-1, keepdim=True) / n_valid

        # 2. Masked Variance across valid frames
        diff = (features - mean) * mask
        var = (diff ** 2).sum(dim=-1, keepdim=True) / n_valid

        # 3. Standard Deviation with stability epsilon
        std = torch.sqrt(var + eps)

        # 4. Normalized features with zero-masked padding
        return (diff / std) * mask

    def compute_lfcc_valid_frames(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        """
        Calculates the exact number of valid temporal frames after LFCC STFT before CNN pooling.

        Args:
            audio_lengths: 1D tensor of raw sample lengths for each item in the batch.

        Returns:
            1D tensor of valid LFCC frame counts, bounded below by 1.
        """
        num_frames = (audio_lengths // self.hop_length) + 1
        return torch.clamp(num_frames, min=1)

    def forward(
        self,
        x: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extracts LFCC feature maps from raw waveform tensor.

        Args:
            x: Raw audio waveform tensor of shape [B, T_samples] or [T_samples].
               Expected to be float32 in [-1.0, 1.0].
            audio_lengths: Optional 1D tensor [B] containing true sample lengths before padding.

        Returns:
            Feature map tensor of shape [B, 1, 60, T_frames] (or [B, 1, 20, T_frames] if with_deltas=False).
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)

        # torchaudio LFCC requires float32 (or float64)
        if x.dtype != torch.float32:
            x = x.float()

        # Compute valid temporal frames if lengths provided
        valid_frames = None
        if audio_lengths is not None:
            valid_frames = self.compute_lfcc_valid_frames(audio_lengths)

        # Extract static LFCC: [B, n_lfcc, T_frames]
        static_lfcc = self.lfcc_transform(x)

        # Optional CMVN before deltas
        if self.use_cmvn and self.cmvn_mode == "before_deltas":
            static_lfcc = self.apply_cmvn(static_lfcc, valid_frames=valid_frames, eps=self.cmvn_eps)

        if self.with_deltas:
            # First derivative (velocity / delta)
            delta = torchaudio.functional.compute_deltas(static_lfcc)
            # Second derivative (acceleration / delta-delta)
            delta_delta = torchaudio.functional.compute_deltas(delta)
            # Concatenate along cepstral coefficient dimension: [B, 60, T_frames]
            features = torch.cat([static_lfcc, delta, delta_delta], dim=1)
        else:
            features = static_lfcc

        # Optional CMVN after deltas (normalizes all 60 channels to unit variance)
        if self.use_cmvn and self.cmvn_mode == "after_deltas":
            features = self.apply_cmvn(features, valid_frames=valid_frames, eps=self.cmvn_eps)

        # Add channel dimension for 2D CNN: [B, 1, 60, T_frames]
        features = features.unsqueeze(1)
        return features

    def compute_cnn_valid_frames(
        self,
        audio_lengths: torch.Tensor,
        num_downsample_stages: int = 4
    ) -> torch.Tensor:
        """
        Calculates the exact number of valid temporal frames after LFCC and N downsampling stages.

        Args:
            audio_lengths: 1D tensor of raw sample lengths for each item in the batch.
            num_downsample_stages: Number of 2x downsampling pooling stages in the CNN.

        Returns:
            1D tensor of valid frame counts after CNN downsampling, bounded below by 1.
        """
        # torchaudio Spectrogram with center=True yields: T_frames = (L // hop_length) + 1
        num_frames = self.compute_lfcc_valid_frames(audio_lengths)

        # Each MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) downsamples floor(T / 2)
        downsampled = num_frames
        for _ in range(num_downsample_stages):
            downsampled = downsampled // 2

        # Ensure at least 1 valid frame
        return torch.clamp(downsampled, min=1)
