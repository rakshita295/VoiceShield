# VoiceShield — Complete ML Repository Cleanup & Mobile Integration Report

**Date:** November 17, 2026  
**Project:** VoiceShield-ML  
**Deployment Target:** Rakshita's Mobile Application (`https://github.com/rakshita295/VoiceShield`)  
**Status:** Complete & Verified  

---

## 1. Executive Summary

The `VoiceShield-ML` repository has undergone a complete architectural audit, cleanup, and restructuring. All obsolete pretrained models (PellaV2, W2V2-AASIST, Wav2Vec2-XLS-R 300M, and fairseq baselines) have been removed from active code paths, reclaiming **over 2.64 GB of disk space**. 

Our custom neural network architecture (**LFCC-CMVN-CNN-BiLSTM-Attention**, 1.74M parameters) is now the sole active model in the repository. Both the original V1 baseline and the selected V2 deployment candidate checkpoints have been verified with cryptographic SHA-256 hashes before and after cleanup, confirming **zero weight modification or degradation**.

The project has been fully prepared for on-device mobile integration with Rakshita's React Native / Expo application via ONNX Runtime Mobile, with full numerical parity verified between PyTorch and ONNX Runtime.

---

## 2. Summary of Cleanup Actions

### 2.1 Pretrained Model Files & Binaries Removed
| File / Directory | Size Reclaimed | Description |
| :--- | :--- | :--- |
| `models/w2v2_aasist/LA_model.pth` | 1.27 GB | Heavy pretrained AASIST checkpoint |
| `models/wav2vec2-xls-r-300m/` | 1.27 GB | Full 300M parameter HuggingFace XLS-R model |
| `experiments/SSL_Anti-spoofing/` | ~100 MB | Fairseq SSL anti-spoofing git submodule/artifacts |
| `src/inference/pellav2.py` | 11.2 KB | PellaV2 pipeline implementation |
| `src/inference/predict.py` | 5.8 KB | Legacy inference script for PellaV2 |
| `src/inference/w2v2_aasist.py` | 16.4 KB | W2V2-AASIST model and feature pipeline |
| `src/inference/predict_w2v2_aasist.py` | 7.9 KB | Legacy inference script for W2V2-AASIST |
| `src/evaluation/benchmark.py` | 14.2 KB | Pretrained model benchmarking script |
| `tests/test_inference.py` | 6.5 KB | Tests exclusively testing PellaV2 inference |
| `tests/test_w2v2_aasist.py` | 8.1 KB | Tests exclusively testing W2V2-AASIST |
| **Total Disk Space Reclaimed** | **> 2.64 GB** | **Cleaned from active repository** |

### 2.2 Historical Documentation Archived
Legacy reports comparing pretrained baselines were relocated to `docs/archive/legacy_baselines/` for historical provenance without cluttering active documentation:
- `docs/archive/legacy_baselines/pellav2_compatibility_report.md`
- `docs/archive/legacy_baselines/w2v2_aasist_implementation_report.md`
- `docs/archive/legacy_baselines/w2v2_aasist_false_positive_investigation.md`
- `docs/archive/legacy_baselines/checkpoint_verification.md`
- `docs/archive/legacy_baselines/model_comparison.md`
- `docs/archive/legacy_baselines/model_feasibility_comparison.md`

### 2.3 Source Code Refactoring & Modularization
1. **`src/preprocessing/audio.py`**: Refactored to provide clean, robust audio loading, channel conversion, and resampling without depending on external pretrained preprocessors.
2. **`src/inference/__init__.py`**: Exports clean unified entry points: `load_detector`, `predict_file`, and `preprocess_audio`.
3. **`src/evaluation/`**: Cleaned of pretrained references; metrics computation (`eer`, `roc_auc`, `min_tdcf`, `evaluate_at_threshold`) remains fully intact.
4. **`tests/test_evaluation.py`**: Cleaned of pretrained benchmark tests while preserving all 4 statistical evaluation tests.

---

## 3. Custom Model Preservation & Hash Verification

Both trained models were verified before and after cleanup using SHA-256 hashing.

### 3.1 Cryptographic Hash Verification
| Model | Path | Checkpoint Epoch | SHA-256 Hash | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **V1 Baseline** | `experiments/custom_lfcc_v1/best_model.pt` | Epoch 10 | `63EEEA2FAB81601AD6706D2990A198C396A87C82109407A8EC8BE4A5586441ED` | **MATCH (Identical)** |
| **V2 Deployment** | `experiments/custom_lfcc_cmvn_v1/best_model.pt` | Epoch 3 | `FDF594FC2638A4E3198A94DB49E757E438F4DC472126F03504FAAC09FF61C86F` | **MATCH (Identical)** |

### 3.2 Independent Loading & Architecture Verification
Both checkpoints were loaded independently and confirmed operational:
- **V1:** 1,736,706 parameters, `use_cmvn=False`, Operating Threshold $\tau = +11.875000$.
- **V2:** 1,736,706 parameters, `use_cmvn=True` (mode: `after_deltas`, $\epsilon=10^{-6}$), Operating Threshold $\tau = -2.417969$.

---

## 4. Test Suite Transition & Smoke Verification

### 4.1 Test Suite Status
The test suite transitioned from 27 tests (including 9 obsolete pretrained tests) to **18 active, focused tests**:
- **CMVN & Normalization (`tests/test_cmvn.py`):** 8 passed.
  - Zero mean / unit variance across channels
  - Before vs. after deltas modes
  - Constant feature stability with epsilon
  - Padded batch masking invariance
  - Training vs. inference parity
  - Output tensor dimensions `[B, 1, 60, T]`
  - V1 backward compatibility (`use_cmvn=False`)
  - End-to-end gradient backpropagation
- **Statistical Evaluation (`tests/test_evaluation.py`):** 4 passed.
  - DET curve & EER computation
  - ROC-AUC bounds
  - Operating point thresholding (FAR, FRR, Acc, F1)
  - Minimum normalized t-DCF computation
- **Model Architecture & Inference (`tests/test_lfcc_model.py`):** 6 passed.
  - Torchaudio LFCC extraction
  - Parameter count verification (1,736,706)
  - Forward and backward pass
  - Mask-aware temporal attention pooling
  - Differential score computation ($s = \text{logit}_1 - \text{logit}_0$)
  - Variable-length batch collation

**Test Result:** `18 passed in 5.11s` (**100% Pass Rate**).

### 4.2 Smoke Test Execution
Running `src/inference/predict_lfcc.py` with V2 on reference audio samples:
- `datasets/samples/genuine_human.wav`:
  - Differential Score: `+14.8021` (Threshold $\tau = -2.417969$)
  - Prediction: **`GENUINE_HUMAN`** (Confidence: 100.00%)
  - Preprocessing + Inference Latency: **5.8 ms**
- `datasets/samples/synthetic_ai.wav`:
  - Differential Score: `-8.9904` (Threshold $\tau = -2.417969$)
  - Prediction: **`AI_GENERATED`** (Confidence: 99.99%)
  - Preprocessing + Inference Latency: **5.2 ms**

---

## 5. Dependency & Configuration Cleanup

### 5.1 `requirements.txt`
- **Purged:** `transformers>=4.40.0`, `huggingface_hub>=0.20.0`
- **Retained Core:** `torch>=2.2.0`, `torchaudio>=2.2.0`, `scipy>=1.12.0`, `numpy>=1.26.0`, `soundfile>=0.12.1`, `pytest>=8.0.0`
- **Added for Deployment:** `onnx>=1.16.0`, `onnxruntime>=1.18.0`, `onnxscript>=0.7.0`

### 5.2 `.gitignore`
Updated with strict rules to guarantee clean repository state:
- Excludes external datasets (`datasets/*`, except `datasets/samples/` reference audio).
- Excludes virtual environments (`.venv/`, `venv/`).
- Excludes heavy model checkpoints and weights (`*.pt`, `*.pth`, `*.onnx`, `models/*`).
- Excludes intermediate epoch checkpoints (`checkpoint_epoch_*.pt`).
- Tracks metadata, configs, and evaluation JSONs (`!experiments/**/*.json`).
- Excludes secrets, API keys, credentials, and IDE cache files.

---

## 6. ONNX Export & Mobile Parity Verification

### 6.1 Export Details
- **Script:** `scripts/export_onnx.py`
- **Output File:** `models/voiceshield_v2.onnx`
- **Model Size:** `6.95 MB` (6,953,979 bytes)
- **Opset Version:** `14`
- **Input Tensor:** `lfcc_features` $\in \mathbb{R}^{B \times 1 \times 60 \times T}$ (dynamic $B, T$)
- **Output Tensors:** `logits` $\in \mathbb{R}^{B \times 2}$, `score` $\in \mathbb{R}^B$

### 6.2 Numerical Parity Check
Comparing PyTorch FP32 forward pass against ONNX Runtime CPU execution:
- **Max Absolute Difference in Logits:** $1.19 \times 10^{-6}$
- **Max Absolute Difference in Score:** $1.91 \times 10^{-6}$
- **Status:** **PASSED** (well below $1.0 \times 10^{-4}$ tolerance).

---

## 7. Active Clean Repository Structure

```
VoiceShield-ML/
│
├── .gitignore                          # Clean ignore rules (no weights/datasets)
├── requirements.txt                    # Minimal ML + ONNX dependencies
├── README.md                           # Complete model architecture & results
│
├── datasets/
│   ├── .gitkeep
│   └── samples/                        # Verified test audio clips
│       ├── genuine_human.wav           # Ground-truth bonafide speech
│       └── synthetic_ai.wav            # Ground-truth spoofed speech
│
├── docs/
│   ├── repository_cleanup_audit.md     # Pre-cleanup file disposition audit
│   ├── repository_cleanup_report.md    # This report
│   ├── mobile_integration_plan.md      # Detailed React Native / Expo integration guide
│   ├── custom_lfcc_model_architecture.md
│   ├── failure_analysis_v1.md
│   ├── experiment_cmvn_v1.md
│   ├── final_evaluation_cmvn_v2.md
│   └── archive/
│       └── legacy_baselines/           # Archived PellaV2 & W2V2-AASIST reports
│
├── experiments/
│   ├── custom_lfcc_v1/                 # V1 Baseline (preserved locally)
│   │   ├── best_model.pt               (SHA-256 verified)
│   │   ├── training_config.json
│   │   └── training_history.json
│   └── custom_lfcc_cmvn_v1/            # V2 Selected Candidate (preserved locally)
│       ├── best_model.pt               (SHA-256 verified)
│       ├── training_config.json
│       ├── training_history.json
│       └── final_evaluation_results.json
│
├── models/
│   ├── .gitkeep
│   └── voiceshield_v2.onnx             # Exported ONNX model (6.95 MB)
│
├── scripts/
│   ├── export_onnx.py                  # PyTorch -> ONNX export & verification
│   └── investigate_failure_modes.py   # Spectral feature analysis script
│
├── src/
│   ├── features/
│   │   ├── __init__.py
│   │   └── lfcc.py                     # LFCC + Utterance CMVN module
│   ├── models/
│   │   ├── __init__.py
│   │   └── lfcc_cnn_bilstm.py          # Custom CNN-BiLSTM-Attention network
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── audio.py                    # Audio conditioning & resampling
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predict_lfcc.py             # Inference pipeline & CLI
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py                  # EER, ROC-AUC, min t-DCF, confusion matrix
│
└── tests/
    ├── test_cmvn.py                    # 8 tests for CMVN normalization
    ├── test_lfcc_model.py              # 6 tests for model architecture & attention
    └── test_evaluation.py              # 4 tests for statistical evaluation metrics
```

---

## 8. Git & Remote Status Report

- **Git Status:** The local workspace `VoiceShield-ML` has been audited.
- **Data & Binaries:** Checkpoint weights (`*.pt`, `*.pth`) and dataset files are strictly ignored in `.gitignore`.
- **Preparation for Sharing:** 
  - Source code, tests, documentation, and configuration are completely decoupled from large model binaries and datasets.
  - A dedicated local branch `ml-integration` is ready to be committed.
- **Safety Boundary:** No remote git operations (push, force-push, or remote additions) will be executed without explicit user instruction.
