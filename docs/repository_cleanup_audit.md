# VoiceShield — Complete Machine Learning Repository Cleanup Audit

**Audit Date:** November 17, 2026  
**Status:** Audit Complete — Pending Cleanup Execution  
**Target Repository:** `VoiceShield-ML`  
**Deployment Target:** Rakshita's Mobile Application (`https://github.com/rakshita295/VoiceShield`)  

---

## 1. Executive Summary

This audit assesses every directory, source file, model weight, test, document, and dependency across `VoiceShield-ML`. 

Our project has successfully completed the development, training, and held-out evaluation of our custom **LFCC-CMVN-CNN-BiLSTM-Attention** model:
- **Selected Deployment Model**: `experiments/custom_lfcc_cmvn_v1/best_model.pt` (Epoch 3, 1,736,706 parameters, ~6.95 MB FP32 weights, Eval EER: 8.736%, Accuracy: 85.70%).
- **Research Baseline**: `experiments/custom_lfcc_v1/best_model.pt` (Epoch 10, 1,736,706 parameters, Eval EER: 9.341%).

All upstream foundation model experiments (PellaV2 Revision 1 and W2V2-AASIST based on `wav2vec2-xls-r-300m`, 315M+ parameters) are obsolete for edge deployment. They occupy **over 2.64 GB of disk storage**, require heavy dependencies (`transformers`, `huggingface_hub`, legacy fairseq artifacts), and cause confusion regarding the repository's production target.

---

## 2. Comprehensive Inventory & Categorization

| Category | Component / Directory | Active or Obsolete | Action Plan |
|---|---|---|---|
| **1. Custom Model Implementation** | `src/models/lfcc_cnn_bilstm.py`, `src/models/__init__.py` | **Active (Core)** | **KEEP** |
| **2. Feature Extraction** | `src/features/lfcc.py`, `src/features/__init__.py` | **Active (Core)** | **KEEP** |
| **3. Inference Pipelines** | `src/inference/predict_lfcc.py`<br>`src/inference/pellav2.py`<br>`src/inference/w2v2_aasist.py`<br>`src/inference/predict.py`<br>`src/inference/predict_w2v2_aasist.py` | Active<br>Obsolete<br>Obsolete<br>Obsolete<br>Obsolete | **KEEP** `predict_lfcc.py`<br>**REMOVE** `pellav2.py`<br>**REMOVE** `w2v2_aasist.py`<br>**REMOVE** `predict.py`<br>**REMOVE** `predict_w2v2_aasist.py` |
| **4. Preprocessing** | `src/preprocessing/audio.py`, `src/preprocessing/__init__.py` | Mixed (Pella/AASIST specific) | **REFACTOR & KEEP**: Retain clean shared audio loading for LFCC; purge obsolete Pella/AASIST pads. |
| **5. Dataset Handling** | `src/datasets/asvspoof2019.py`, `src/datasets/__init__.py` | **Active (Core)** | **KEEP** |
| **6. Training Engine** | `src/training/train.py`, `src/training/trainer.py`, `src/training/__init__.py` | **Active (Core)** | **KEEP** |
| **7. Evaluation & Auditing** | `src/evaluation/metrics.py`<br>`src/evaluation/evaluate_asvspoof2019_eval.py`<br>`src/evaluation/audit_dev_predictions.py`<br>`src/evaluation/benchmark.py` | Active<br>Active<br>Active<br>Obsolete (AASIST 2021) | **KEEP** `metrics.py`<br>**KEEP** `evaluate_asvspoof2019_eval.py`<br>**KEEP** `audit_dev_predictions.py`<br>**REMOVE** `benchmark.py` |
| **8. Pretrained Model Weights** | `models/w2v2_aasist/LA_model.pth` (1.27 GB)<br>`models/wav2vec2-xls-r-300m/` (1.27 GB) | Obsolete<br>Obsolete | **REMOVE** (Frees 2.54 GB) |
| **9. Legacy Fairseq / SSL Artifacts** | `experiments/SSL_Anti-spoofing/` (~100 MB, fairseq clone, score txts) | Obsolete | **REMOVE** |
| **10. Custom Checkpoints (V1 & V2)**| `experiments/custom_lfcc_v1/`<br>`experiments/custom_lfcc_cmvn_v1/` | **Active (Mandatory)** | **PRESERVE 100%**: Keep `best_model.pt`, evaluation results, history, and configs. |
| **11. Test Suite** | `tests/test_cmvn.py`<br>`tests/test_lfcc_model.py`<br>`tests/test_evaluation.py`<br>`tests/test_inference.py` (PellaV2)<br>`tests/test_w2v2_aasist.py` (W2V2-AASIST) | Active<br>Active<br>Active (Refactor)<br>Obsolete<br>Obsolete | **KEEP** `test_cmvn.py`<br>**KEEP** `test_lfcc_model.py`<br>**KEEP & REFACTOR** `test_evaluation.py`<br>**REMOVE** `test_inference.py`<br>**REMOVE** `test_w2v2_aasist.py` |
| **12. Dependencies** | `requirements.txt` | Has unused packages (`transformers`, `huggingface_hub`) | **CLEAN & UPDATE**: Remove pretrained dependencies; add `onnx`, `onnxruntime`. |
| **13. Documentation** | Active custom reports vs. legacy XLS-R reports | Mixed | **ARCHIVE** legacy reports to `docs/archive/legacy_baselines/`; keep custom reports active. |

---

## 3. Detailed File-by-File Disposition Matrix

### 3.1 Pretrained Model Implementations (To Remove)

| File Path | Size | Purpose | References / Imports | Recommended Action | Justification |
|---|---|---|---|:---:|---|
| `src/inference/pellav2.py` | 7.95 KB | PellaV2 Revision 1 model wrapper around HuggingFace Wav2Vec2 | `src/inference/predict.py`, `src/inference/__init__.py`, `tests/test_inference.py` | **REMOVE** | PellaV2 is a 317M parameter model unsuitable for edge deployment. |
| `src/inference/predict.py` | 6.96 KB | CLI prediction runner for PellaV2 | None outside itself | **REMOVE** | Replaced by `src/inference/predict_lfcc.py`. |
| `src/inference/w2v2_aasist.py` | 21.61 KB | W2V2-AASIST model architecture with graph attention (AASIST) | `src/inference/predict_w2v2_aasist.py`, `src/inference/__init__.py`, `tests/test_w2v2_aasist.py` | **REMOVE** | 315M parameter server baseline. Replaced by custom 1.74M model. |
| `src/inference/predict_w2v2_aasist.py` | 9.33 KB | CLI prediction runner for W2V2-AASIST | None outside itself | **REMOVE** | Replaced by `src/inference/predict_lfcc.py`. |
| `src/evaluation/benchmark.py` | 7.40 KB | Benchmark runner for ASVspoof 2021 LA using W2V2-AASIST and fairseq files | `tests/test_evaluation.py` | **REMOVE** | Evaluates legacy text score files from `experiments/SSL_Anti-spoofing`. |

### 3.2 Pretrained Checkpoints & Large Data Dumps (To Remove)

| File Path / Directory | Size | Purpose | Recommended Action | Justification |
|---|---|---|:---:|---|
| `models/w2v2_aasist/LA_model.pth` | 1.27 GB (1,271,633,441 bytes) | Pretrained AASIST + XLS-R checkpoint | **REMOVE** | Massive obsolete binary; frees 1.27 GB. |
| `models/wav2vec2-xls-r-300m/pytorch_model.bin` | 1.27 GB (1,269,737,156 bytes) | HuggingFace XLS-R 300M weights | **REMOVE** | Massive obsolete binary; frees 1.27 GB. |
| `models/wav2vec2-xls-r-300m/config.json` | 1.57 KB | XLS-R HuggingFace configuration | **REMOVE** | Associated with deleted binary. |
| `models/wav2vec2-xls-r-300m/preprocessor_config.json` | 211 bytes | XLS-R preprocessor configuration | **REMOVE** | Associated with deleted binary. |
| `experiments/SSL_Anti-spoofing/` | ~100 MB (1,300+ files) | Fairseq git checkout, 2021 LA trial metadata, legacy score dumps | **REMOVE** | Unused legacy artifacts from preliminary research. |

### 3.3 Obsolete Pretrained Tests (To Remove or Refactor)

| File Path | Purpose | Recommended Action | Justification |
|---|---|:---:|---|
| `tests/test_inference.py` | Tests PellaV2 architecture and prediction | **REMOVE** | Tests deleted PellaV2 code. |
| `tests/test_w2v2_aasist.py` | Tests W2V2-AASIST architecture and `LA_model.pth` | **REMOVE** | Tests deleted W2V2-AASIST code. |
| `tests/test_evaluation.py` | Tests EER, AUC, min t-DCF, and `benchmark.py` | **REFACTOR & KEEP** | Retain 4 tests for synthetic DET/EER, ROC-AUC, threshold metrics, and min t-DCF. Remove only `test_benchmark_data_partitioning_and_calibration` which depends on deleted `benchmark.py`. |

### 3.4 Custom Model Core Components (To Preserve 100%)

| File Path | Purpose | Recommended Action |
|---|---|:---:|
| `src/features/lfcc.py` | Native Torchaudio LFCC + Utterance-Level CMVN + padding-aware masking | **KEEP** |
| `src/features/__init__.py` | Exports LFCCFeatureExtractor | **KEEP** |
| `src/models/lfcc_cnn_bilstm.py` | Custom LFCC-CNN-BiLSTM-Attention detector (1.74M parameters) | **KEEP** |
| `src/models/__init__.py` | Exports model and temporal attention pooling | **KEEP** |
| `src/inference/predict_lfcc.py` | Production inference CLI supporting both V1 and V2 models | **KEEP** |
| `src/datasets/asvspoof2019.py` | ASVspoof 2019 LA Dataset loader and variable-length collator | **KEEP** |
| `src/datasets/__init__.py` | Exports dataset utilities | **KEEP** |
| `src/training/train.py` | Model training CLI | **KEEP** |
| `src/training/trainer.py` | Core training engine with mixed-precision, early stopping, and metrics | **KEEP** |
| `src/training/__init__.py` | Exports trainer components | **KEEP** |
| `src/evaluation/metrics.py` | Pure numerical evaluation functions (EER, AUC, t-DCF, confusion matrix) | **KEEP** |
| `src/evaluation/evaluate_asvspoof2019_eval.py` | Official ASVspoof 2019 LA held-out evaluation runner | **KEEP** |
| `src/evaluation/audit_dev_predictions.py` | Development set prediction audit script | **KEEP** |
| `tests/test_lfcc_model.py` | 6 unit tests for custom LFCC model, attention masking, collator | **KEEP** |
| `tests/test_cmvn.py` | 8 unit tests for utterance-level CMVN mathematical properties | **KEEP** |
| `datasets/samples/genuine_human.wav` | Reference audio sample for testing | **KEEP** |
| `datasets/samples/synthetic_ai.wav` | Reference audio sample for testing | **KEEP** |
| `datasets/samples/genuine_human.flac` | Reference audio sample for testing | **KEEP** |

### 3.5 Checkpoint Verification (Mandatory Preservation)

| Checkpoint Path | Description | Checkpoint SHA-256 Hash | Status |
|---|---|---|:---:|
| `experiments/custom_lfcc_v1/best_model.pt` | V1 Baseline Checkpoint (Epoch 10) | `63EEEA2FAB81601AD6706D2990A198C396A87C82109407A8EC8BE4A5586441ED` | **PRESERVED** |
| `experiments/custom_lfcc_cmvn_v1/best_model.pt` | V2 Selected Checkpoint (Epoch 3) | `FDF594FC2638A4E3198A94DB49E757E438F4DC472126F03504FAAC09FF61C86F` | **PRESERVED** |

All accompanying JSON history files, prediction arrays (`eval_predictions.npz`), and telemetry in both experiment folders will be preserved.

### 3.6 Documentation Reorganization

Historical documents comparing obsolete baselines will be moved to `docs/archive/legacy_baselines/` to prevent confusion:
- `docs/pellav2_compatibility_report.md` $\to$ `docs/archive/legacy_baselines/`
- `docs/w2v2_aasist_implementation_report.md` $\to$ `docs/archive/legacy_baselines/`
- `docs/w2v2_aasist_false_positive_investigation.md` $\to$ `docs/archive/legacy_baselines/`
- `docs/checkpoint_verification.md` $\to$ `docs/archive/legacy_baselines/`
- `docs/model_comparison.md` $\to$ `docs/archive/legacy_baselines/`
- `docs/model_feasibility_comparison.md` $\to$ `docs/archive/legacy_baselines/`

Active documents explaining our custom architecture and evaluation will remain in `docs/`:
- `docs/custom_lfcc_model_architecture.md`
- `docs/asvspoof_dataset_audit.md`
- `docs/checkpoint_and_metric_audit.md`
- `docs/failure_analysis_v1.md`
- `docs/final_evaluation_report.md` (V1 baseline evaluation)
- `docs/experiment_cmvn_v1.md`
- `docs/training_report_cmvn_v2.md`
- `docs/final_evaluation_cmvn_v2.md` (V2 held-out evaluation)
- `docs/repository_cleanup_audit.md` (This document)
- `docs/repository_cleanup_report.md`
- `docs/mobile_integration_plan.md`

---

## 4. Summary of Space Reclaimed

| Component Removed | Disk Space Reclaimed |
|---|---|
| `models/w2v2_aasist/LA_model.pth` | 1.27 GB |
| `models/wav2vec2-xls-r-300m/pytorch_model.bin` | 1.27 GB |
| `experiments/SSL_Anti-spoofing/` | ~100 MB |
| **Total Reclaimed Space** | **~2.64 GB (>80% of repository)** |

---

## 5. Next Steps

Upon review of this audit:
1. Execute file removals for obsolete pretrained source files, model binaries, and legacy tests.
2. Refactor `src/preprocessing/audio.py` and `src/inference/__init__.py` to provide a clean interface for the custom detector.
3. Update `requirements.txt` to remove `transformers` and `huggingface_hub`.
4. Update `.gitignore`.
5. Update `README.md`.
6. Run the remaining unit tests (verifying 100% pass rate).
7. Prepare `docs/mobile_integration_plan.md` and `docs/repository_cleanup_report.md`.
