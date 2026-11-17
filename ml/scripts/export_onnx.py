"""
VoiceShield Model ONNX Exporter.

Exports the trained PyTorch LFCC-CNN-BiLSTM-Attention classifier backbone
to an optimized ONNX model formatted for mobile deployment with ONNX Runtime Mobile.

Verification:
- Validates model structure with onnx.checker
- Executes inference parity checks against PyTorch using ONNX Runtime
- Verifies dynamic temporal frame lengths for variable-length speech input
"""

import argparse
from pathlib import Path
import sys
from typing import Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

from src.inference.predict_lfcc import load_detector


class VoiceShieldONNXWrapper(nn.Module):
    """
    Inference-ready wrapper for the CNN-BiLSTM-Attention classifier backbone.

    Accepts preprocessed LFCC+CMVN features:
        Shape: [B, 1, 60, T_frames] (float32)
    Returns:
        logits: [B, 2] where [:, 0] is Spoof, [:, 1] is Bonafide
        score:  [B] where score = logit_bonafide - logit_spoof
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.block1 = model.block1
        self.block2 = model.block2
        self.block3 = model.block3
        self.block4 = model.block4
        self.lstm = model.lstm
        self.attention = model.attention
        self.classifier = model.classifier

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # CNN Feature Extraction: [B, 256, 3, T_cnn]
        c = self.block1(features)
        c = self.block2(c)
        c = self.block3(c)
        c = self.block4(c)

        batch_size, channels, freq_dim, time_dim = c.shape
        seq = c.permute(0, 3, 1, 2).contiguous().view(batch_size, time_dim, channels * freq_dim)

        # BiLSTM Sequence Processing: [B, T_cnn, 256]
        lstm_out, _ = self.lstm(seq)

        # Temporal Attention Pooling: [B, 256]
        context, _ = self.attention(lstm_out)

        # Classifier Head: [B, 2]
        logits = self.classifier(context)

        # Differential log-likelihood ratio: Bonafide (idx 1) - Spoof (idx 0)
        score = logits[:, 1] - logits[:, 0]

        return logits, score


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 14,
    verify: bool = True,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from: {checkpoint_path}")
    model, threshold, config = load_detector(str(checkpoint_path), device="cpu")
    model.eval()

    print(f"Model parameters: {model.count_parameters()['total_parameters']:,}")
    print(f"Feature CMVN enabled: {model.use_cmvn} (mode: {model.cmvn_mode})")
    print(f"Operating decision threshold: {threshold:.6f}")

    wrapper = VoiceShieldONNXWrapper(model)
    wrapper.eval()

    # Create dummy input: 1 utterance, 1 channel, 60 LFCC bins, 400 temporal frames (~4 seconds)
    dummy_input = torch.randn(1, 1, 60, 400, dtype=torch.float32)

    with torch.no_grad():
        pt_logits, pt_score = wrapper(dummy_input)

    print(f"Exporting ONNX model to: {output_path} (opset {opset_version})...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path),
        input_names=["lfcc_features"],
        output_names=["logits", "score"],
        dynamic_axes={
            "lfcc_features": {0: "batch_size", 3: "time_frames"},
            "logits": {0: "batch_size"},
            "score": {0: "batch_size"},
        },
        opset_version=opset_version,
        dynamo=False,
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Export complete. Model size: {file_size_mb:.2f} MB")

    if verify:
        print("Running ONNX structural check...")
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model structure: VALID")

        print("Testing inference parity via ONNX Runtime...")
        session = ort.InferenceSession(str(output_path))
        ort_outs = session.run(None, {"lfcc_features": dummy_input.numpy()})
        ort_logits, ort_score = ort_outs

        diff_logits = float(np.max(np.abs(pt_logits.numpy() - ort_logits)))
        diff_score = float(np.max(np.abs(pt_score.numpy() - ort_score)))
        print(f"Max absolute logits diff: {diff_logits:.2e}")
        print(f"Max absolute score diff:  {diff_score:.2e}")

        assert diff_logits < 1e-4, f"Logits parity failure (diff={diff_logits})"
        assert diff_score < 1e-4, f"Score parity failure (diff={diff_score})"
        print("ONNX Runtime parity: PASSED (tolerance < 1e-4)")

        # Verify dynamic time frames
        print("Verifying dynamic time frames across variable lengths...")
        for t in [100, 250, 400, 750]:
            test_feat = np.random.randn(1, 1, 60, t).astype(np.float32)
            out = session.run(None, {"lfcc_features": test_feat})
            assert out[0].shape == (1, 2)
            assert out[1].shape == (1,)
        print("Dynamic time frames verification: PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export VoiceShield model to ONNX format")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/custom_lfcc_cmvn_v1/best_model.pt",
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/voiceshield_v2.onnx",
        help="Destination path for exported ONNX model",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip numerical verification with ONNX Runtime",
    )
    args = parser.parse_args()

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
