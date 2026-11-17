# VoiceShield — Machine Learning Subsystem

This directory contains the machine learning core of **VoiceShield**, an on-device deepfake speech detection engine designed to detect synthetic, cloned, and replay voice attacks.

The model is built entirely from scratch and trained on the ASVspoof 2019 Logical Access (LA) benchmark. It operates without heavy pretrained audio foundation models (such as XLS-R 300M or Wav2Vec2), maintaining an ultra-light footprint of **1,736,706 parameters (~6.95 MB FP32)** suitable for mobile execution.

---

## 1. Architecture Overview

```
Raw Audio (16 kHz Mono)
         │
         ▼
[ torchaudio LFCC Feature Extractor ]
    • 20 static coefficients (512 FFT, 25ms win, 10ms hop, Hann)
    • 20 delta coefficients (velocity, N=2)
    • 20 delta-delta coefficients (acceleration, N=2)
    • Output shape: [Batch, 1, 60, Time]
         │
         ▼
[ Utterance-Level CMVN ]
    • Normalizes cepstral feature distribution per utterance
    • Per-channel zero mean, unit variance (eps = 1e-6)
         │
         ▼
[ 4-Stage 2D-CNN Backbone ]
    • Block 1: Conv2d(1->32, 3x3) + BatchNorm2d + LeakyReLU(0.2) + MaxPool2d(2x2)
    • Block 2: Conv2d(32->64, 3x3) + BatchNorm2d + LeakyReLU(0.2) + MaxPool2d(2x2)
    • Block 3: Conv2d(64->128, 3x3) + BatchNorm2d + LeakyReLU(0.2) + MaxPool2d(2x2)
    • Block 4: Conv2d(128->256, 3x3) + BatchNorm2d + LeakyReLU(0.2) + MaxPool2d(2x2)
         │
         ▼
[ 2-Layer Bidirectional LSTM ]
    • Hidden size: 128 per direction (256 bidirectional)
    • Models temporal acoustic dependencies across frames
         │
         ▼
[ Mask-Aware Temporal Self-Attention Pooling ]
    • Computes frame importance weights: alpha_t = softmax(v^T tanh(W h_t + b))
    • Aggregates temporal embeddings into a fixed 256-dim context vector
         │
         ▼
[ Classification Head ]
    • Linear(256 -> 64) -> LeakyReLU(0.2) -> Dropout(0.5) -> Linear(64 -> 2)
    • Logits: [Spoof (idx 0), Bonafide (idx 1)]
    • Differential Score: score = logit_bonafide - logit_spoof
```

---

## 2. Benchmark Performance (ASVspoof 2019 LA)

| Metric | V1 Baseline | V2 (Utterance CMVN) | Delta / Improvement |
| :--- | :--- | :--- | :--- |
| **Evaluation EER** | 9.341% | **8.736%** | **-0.605% absolute (-6.48% relative)** |
| **Evaluation ROC-AUC** | 0.96830 | **0.96327** | High discriminative power |
| **Accuracy (Dev Threshold)**| 81.33% | **85.70%** | **+4.37% absolute improvement** |
| **FAR (Dev Threshold)** | 18.337% | **15.660%** | **-2.677% absolute reduction** |
| **FRR (Dev Threshold)** | 0.857% | **2.502%** | Balanced trade-off |
| **Operating Threshold ($\tau$)**| `+11.875000` | `-2.417969` | Calibrated on development split |
| **Parameter Count** | 1,736,706 | 1,736,706 | Identical lightweight architecture |
| **Model Size (FP32)** | ~6.95 MB | ~6.95 MB | Fits in low-end device RAM |

---

## 3. Directory Layout

```
ml/
├── src/
│   ├── features/
│   │   └── lfcc.py                     # LFCC extraction + Utterance CMVN
│   ├── models/
│   │   └── lfcc_cnn_bilstm.py          # Custom CNN-BiLSTM-Attention network
│   ├── preprocessing/
│   │   └── audio.py                    # Audio conditioning & resampling
│   ├── inference/
│   │   └── predict_lfcc.py             # Inference pipeline & CLI
│   ├── evaluation/
│   │   └── metrics.py                  # EER, ROC-AUC, min t-DCF, confusion matrix
│   ├── training/
│   │   ├── train.py                    # Model training entry point
│   │   └── trainer.py                  # PyTorch training loop with early stopping
│   └── datasets/
│       └── asvspoof2019.py             # ASVspoof 2019 LA Dataset loader
│
├── tests/
│   ├── conftest.py                     # sys.path environment configuration
│   ├── test_cmvn.py                    # 8 tests for CMVN normalization
│   ├── test_lfcc_model.py              # 6 tests for model architecture & attention
│   ├── test_evaluation.py              # 4 tests for statistical evaluation metrics
│   └── samples/                        # Sample reference audio clips
│       ├── genuine_human.wav
│       └── synthetic_ai.wav
│
├── scripts/
│   ├── export_onnx.py                  # PyTorch to ONNX exporter with parity verification
│   └── investigate_failure_modes.py   # Spectral failure mode analysis
│
├── docs/
│   ├── mobile_integration_plan.md      # React Native / Expo integration blueprint
│   ├── custom_lfcc_model_architecture.md
│   ├── final_evaluation_cmvn_v2.md
│   └── repository_cleanup_report.md
│
└── requirements.txt                    # ML dependencies
```

---

## 4. Setup & Verification

### 4.1 Install Dependencies
```bash
pip install -r ml/requirements.txt
```

### 4.2 Run Test Suite
All 18 unit tests verify feature extraction, CMVN normalization, attention pooling, and evaluation metrics:
```bash
pytest ml/tests -v
```

### 4.3 Run CLI Inference
```bash
python -m ml.src.inference.predict_lfcc ml/tests/samples/genuine_human.wav ml/tests/samples/synthetic_ai.wav
```

### 4.4 Export ONNX Model for Mobile Deployment
```bash
python ml/scripts/export_onnx.py --checkpoint path/to/best_model.pt --output models/voiceshield_v2.onnx
```

For complete mobile integration specifications (tensor shapes, Expo Native Modules in Kotlin, ONNX Runtime Mobile), refer to [docs/mobile_integration_plan.md](docs/mobile_integration_plan.md).
