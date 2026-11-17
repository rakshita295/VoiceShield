"""
Custom LFCC-CNN-BiLSTM-Attention Deepfake Speech Detection Model.

Architecture:
1. torchaudio LFCC feature extraction (Static + Delta + Delta-Delta = 60 channels)
2. 4 Custom 2D-CNN Blocks (Conv2d + BatchNorm2d + LeakyReLU + MaxPool2d)
3. 2-Layer Bidirectional LSTM (Temporal dependency modeling across frames)
4. Mask-aware Temporal Self-Attention Pooling (Weights frames, ignores zero-padding)
5. Binary Classification Head: Spoof (Index 0) vs. Bonafide (Index 1)

All neural network weights are randomly initialized from scratch.
Zero pretrained models or checkpoints are loaded.
"""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.features.lfcc import LFCCFeatureExtractor


class TemporalAttentionPooling(nn.Module):
    """
    Self-attention pooling over temporal frames with padding mask support.

    Computes:
        e_t = v^T tanh(W h_t + b)
        alpha_t = softmax(e_t, masked for padding)
        context = sum_t (alpha_t * h_t)
    """

    def __init__(self, in_dim: int = 256, attn_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, attn_dim)
        self.energy = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        valid_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Frame embeddings of shape [B, T, in_dim].
            valid_lengths: Optional 1D tensor [B] of valid frame counts per utterance.

        Returns:
            Tuple of:
            - context: Pooled vector of shape [B, in_dim]
            - alpha: Attention weights of shape [B, T]
        """
        # Score computation: [B, T, 1]
        score = self.energy(torch.tanh(self.proj(h))).squeeze(-1)  # [B, T]

        if valid_lengths is not None:
            batch_size, max_t = score.shape
            device = score.device
            mask = torch.arange(max_t, device=device).unsqueeze(0) < valid_lengths.unsqueeze(1)
            # Mask out padding with float16-safe large negative number
            fill_value = -1e4 if score.dtype == torch.float16 else -1e9
            score = score.masked_fill(~mask, fill_value)

        alpha = torch.softmax(score, dim=-1)  # [B, T]

        # Context vector: sum over time
        context = torch.sum(h * alpha.unsqueeze(-1), dim=1)  # [B, in_dim]
        return context, alpha


class LFCC_CNN_BiLSTM(nn.Module):
    """
    Fully custom, randomly initialized deepfake speech detector.

    Args:
        sample_rate: Audio sampling rate (default: 16000).
        n_filter: LFCC filterbank count (default: 20).
        n_lfcc: LFCC cepstral coefficients (default: 20).
        cnn_channels: Tuple of output channels for the 4 CNN blocks.
        lstm_hidden: Hidden size for each direction of the BiLSTM.
        lstm_layers: Number of BiLSTM layers.
        attn_dim: Dimension of attention projection layer.
        dropout: Dropout probability.
        use_cmvn: Whether to apply utterance-level CMVN in LFCC front-end (default: False).
        cmvn_mode: 'after_deltas' (default) or 'before_deltas'.
        cmvn_eps: Epsilon for numerical stability (default: 1e-6).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_filter: int = 20,
        n_lfcc: int = 20,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        cnn_channels: Tuple[int, int, int, int] = (32, 64, 128, 256),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        attn_dim: int = 64,
        dropout: float = 0.5,
        use_cmvn: bool = False,
        cmvn_mode: str = "after_deltas",
        cmvn_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.use_cmvn = use_cmvn
        self.cmvn_mode = cmvn_mode
        self.cmvn_eps = cmvn_eps

        # 1. LFCC Feature Extractor
        self.feature_extractor = LFCCFeatureExtractor(
            sample_rate=sample_rate,
            n_filter=n_filter,
            n_lfcc=n_lfcc,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            with_deltas=True,
            log_lf=True,
            use_cmvn=use_cmvn,
            cmvn_mode=cmvn_mode,
            cmvn_eps=cmvn_eps,
        )

        in_channels = 1
        c1, c2, c3, c4 = cnn_channels

        # 2. Four Custom 2D-CNN Blocks
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        # After 4 downsampling stages along frequency (60 -> 30 -> 15 -> 7 -> 3)
        self.freq_dim_out = 3
        self.cnn_flat_dim = c4 * self.freq_dim_out  # 256 * 3 = 768

        # 3. Two-Layer Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.cnn_flat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_out_dim = lstm_hidden * 2  # Bidirectional -> 256

        # 4. Temporal Attention Pooling
        self.attention = TemporalAttentionPooling(in_dim=self.lstm_out_dim, attn_dim=attn_dim)

        # 5. Binary Classifier Head: [spoof (0), bonafide (1)]
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_out_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

        # Initialize all weights randomly from scratch
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initializes all model weights from scratch using Kaiming normal for Conv/Linear
        and Orthogonal for LSTM recurrent matrices.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
                        # Set forget gate bias to 1.0 for LSTM stability
                        n = param.size(0)
                        param.data[n // 4 : n // 2].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Raw audio waveform tensor of shape [B, T_samples].
            audio_lengths: Optional 1D tensor [B] of sample lengths before zero-padding.
            return_attn: If True, returns attention distribution across frames.

        Returns:
            Tuple of:
            - logits: Tensor of shape [B, 2] where [:, 0] is spoof logit, [:, 1] is bonafide logit.
            - attn_weights: Optional tensor of shape [B, T_cnn] if return_attn=True, else None.
        """
        # 1. LFCC Feature Extraction: [B, 1, 60, T_frames]
        # Torchaudio STFT requires float32
        features = self.feature_extractor(x, audio_lengths=audio_lengths)

        # Calculate valid temporal frames after CNN pooling if lengths are provided
        valid_frames = None
        if audio_lengths is not None:
            valid_frames = self.feature_extractor.compute_cnn_valid_frames(
                audio_lengths, num_downsample_stages=4
            )

        # 2. CNN Backbone: [B, 256, 3, T_cnn]
        c = self.block1(features)
        c = self.block2(c)
        c = self.block3(c)
        c = self.block4(c)

        batch_size, channels, freq_dim, time_dim = c.shape

        # Reshape to temporal sequence for LSTM: [B, T_cnn, channels * freq_dim] = [B, T_cnn, 768]
        seq = c.permute(0, 3, 1, 2).contiguous().view(batch_size, time_dim, channels * freq_dim)

        # 3. Bidirectional LSTM
        if valid_frames is not None:
            # Bound valid frames by current actual time_dim
            clamped_valid = torch.clamp(valid_frames, min=1, max=time_dim)
            packed = pack_padded_sequence(
                seq, clamped_valid.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out_packed, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True, total_length=time_dim)
        else:
            lstm_out, _ = self.lstm(seq)

        # 4. Temporal Attention Pooling: [B, 256]
        context, attn_weights = self.attention(lstm_out, valid_lengths=valid_frames)

        # 5. Classifier Head: [B, 2]
        logits = self.classifier(context)

        if return_attn:
            return logits, attn_weights
        return logits, None

    def compute_score(
        self,
        x: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes the bonafide log-likelihood ratio score: (bonafide_logit - spoof_logit).
        Higher score indicates higher confidence of genuine human speech.

        Args:
            x: Raw audio waveform tensor [B, T_samples].
            audio_lengths: Optional sample lengths.

        Returns:
            1D tensor of differential scores [B].
        """
        logits, _ = self.forward(x, audio_lengths=audio_lengths)
        # Log-likelihood ratio / differential score: bonafide (idx 1) - spoof (idx 0)
        return logits[:, 1] - logits[:, 0]

    def count_parameters(self) -> Dict[str, int]:
        """Returns total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}
