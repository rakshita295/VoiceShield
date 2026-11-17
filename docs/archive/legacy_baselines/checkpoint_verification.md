# Forensic Verification Report: PellaV2 Checkpoint & Weights Status

**Project**: VoiceShield-ML  
**Date of Audit**: 2026-11-16  
**Auditor**: Senior ML Research Engineer (Autonomous Agent)  
**Status**: **CRITICAL DEFECT IDENTIFIED — TRAINED DETECTOR WEIGHTS UNAVAILABLE**

---

## 1. Executive Summary

A forensic audit of the VoiceShield PellaV2 Revision 1 implementation revealed that while the self-supervised foundational backbone (`facebook/wav2vec2-xls-r-300m`, 315M parameters) was successfully downloaded and loaded onto the local NVIDIA RTX 4050 GPU, **the trained PellaV2 Revision 1 anti-spoofing detector weights were NEVER loaded**.

The outputs obtained during GPU smoke testing (e.g., $p_{\text{fake}} = 0.0275$ for genuine speech and $p_{\text{fake}} = 0.0658$ for synthetic speech) were produced by passing raw XLS-R self-supervised representations into an **un-trained, randomly initialized classification head** with uniform layer weighting. **These predictions carry zero statistical validity and must NOT be treated as genuine PellaV2 detection results.**

Furthermore, an exhaustive investigation across Hugging Face commit histories, author repository discussions, independent benchmark repositories, and third-party mirrors confirmed that **the original trained PellaV2 Revision 1 detector checkpoint is permanently unavailable**.

---

## 2. Technical Inspection of `src/inference/pellav2.py`

### 2.1 Initialization of Weights & Classification Head
In `src/inference/pellav2.py` (lines 42–46):

```python
# 1 embedding representation + 24 transformer layers = 25 hidden states
num_layers = self.backbone.config.num_hidden_layers + 1
self.layer_weights = nn.Parameter(torch.zeros(num_layers))
self.head = nn.Linear(self.backbone.config.hidden_size, 1)
```

1. **`layer_weights`**:
   - Initialized to `torch.zeros(25)`: all elements are strictly $0.0$.
   - During the forward pass:
     $$\mathbf{w}_{\text{norm}} = \text{softmax}(\mathbf{0}_{25}) = \left[\frac{1}{25}, \frac{1}{25}, \dots, \frac{1}{25}\right] = [0.04, 0.04, \dots, 0.04]$$
   - It performs an unweighted uniform arithmetic average across all 24 Transformer layers and the embedding layer, rather than utilizing trained layer-importance weights.
2. **Classification `head` (`nn.Linear(1024, 1)`)**:
   - Initialized via PyTorch's default `nn.Linear` initializer (Kaiming uniform):
     $$W \sim \mathcal{U}\left(-\frac{1}{\sqrt{1024}}, +\frac{1}{\sqrt{1024}}\right) = \mathcal{U}(-0.03125, +0.03125), \quad b \sim \mathcal{U}(-0.03125, +0.03125)$$
   - It consists entirely of **pseudo-random, untrained weights**.

### 2.2 Checkpoint Loading Execution Trace
In `src/inference/predict.py`:
```python
parser.add_argument("--checkpoint", type=str, default=None, ...)
```
When running:
```powershell
.\.venv\Scripts\python.exe src/inference/predict.py datasets/samples/genuine_human.wav datasets/samples/synthetic_ai.wav
```
- `args.checkpoint` was `None`.
- In `load_pellav2_revision1(checkpoint_path=args.checkpoint, device=device)`:
  ```python
  if checkpoint_path is not None:
      # This block was NEVER executed!
      is_compat, msg, _ = verify_checkpoint_compatibility(checkpoint_path)
      ...
      model.load_state_dict(state_dict)
  ```
- **`model.load_state_dict()` was NEVER called.**
- The only weights loaded into memory were the self-supervised pre-training weights of `facebook/wav2vec2-xls-r-300m`.

---

## 3. Checkpoint Inventory & Verification Status

| Component | Path / Source | File Size | Actual Status |
| :--- | :--- | :--- | :--- |
| **Backbone Checkpoint** | `models/wav2vec2-xls-r-300m/pytorch_model.bin` | 1,269,737,156 bytes (~1.27 GB) | **VERIFIED & LOADED** (Base SSL XLS-R 300M weights from Meta) |
| **Backbone Config** | `models/wav2vec2-xls-r-300m/config.json` | 1,568 bytes | **VERIFIED** |
| **Trained Detector Checkpoint (Revision 1)** | Intended: `ee0e94b` on `Sadanie/pellav2-audio-deepfake-detector` | Unknown (~1.26 GB) | **MISSING / UNAVAILABLE** (HTTP 404, purged from HF) |
| **Current HF Checkpoint** | `Sadanie/pellav2-audio-deepfake-detector:pellav2_detector.pt` | 2,322,248,702 bytes (~2.32 GB) | **INCOMPATIBLE** (Revision 2, W2V-BERT 2.0 Conformer) |
| **`layer_weights`** | In-memory `model.layer_weights` | 25 parameters | **UNTRAINED** (`torch.zeros`) |
| **Classification `head`** | In-memory `model.head` | 1,025 parameters | **UNTRAINED** (Kaiming uniform random) |

---

## 4. Forensic Investigation of Hugging Face & Community Repositories

### 4.1 Author Repository (`Sadanie/pellav2-audio-deepfake-detector`)
1. **Commit History**:
   - `b668eb4dfe4719665692c73c1d70383680afff6b`: *"pointer to pellav3"* (2026-11-06)
   - `29daea8b04acf2d3f11d1d0da6b01c17fc8a1e4b`: *"pellav2_detector.pt"* (2026-10-25)
   - `808e16a49526178f8e1e6276725dfd14f218ec82`: *"Super-squash branch 'main' using huggingface_hub"* (2026-10-22)
   - The repository was completely squashed into commit `808e16a4`. Git refs report only `refs/heads/main` and zero tags.
   - Historical commit `ee0e94be772066edf2b6289a77b8602de4c52164` returns `HTTP 404: EntryNotFound`. All pre-squash LFS blobs were unreferenced and garbage-collected.

2. **First-Party Author Confirmation (Discussion #2)**:
   Author `Sadanie` (Adan) explicitly answered the community on Hugging Face:
   > *"Backbone: the current pellav2_detector.pt is w2v-BERT 2.0 with a weighted sum over the hidden layers, and the pellav2_infer.py on main loads facebook/w2v-bert-2.0 to match. The XLS-R-300M script was the first release and was replaced on 22 October together with the weights, so please pull the script from main again."*

   This confirms directly from the author that **on 2026-10-22, Revision 1 (XLS-R 300M) was intentionally retired, discarded, and replaced with Revision 2 (W2V-BERT 2.0)**.

### 4.2 Independent Benchmarks (`SpeechAntiSpoofingBenchmarks/Pellav2`)
An independent team evaluating PellaV2 documented this exact discrepancy in their `implementation-notes.md`:
> *"Architecture deviates from the plan row: the plan says 'XLS-R + learned weights over all 25 layers'; the script is w2v-BERT 2.0 (`facebook/w2v-bert-2.0`, 24 conformer layers, 580 M params)... Files: `pellav2_detector.pt` (2.32 GB)..."*

They confirmed that only the 2.32 GB W2V-BERT 2.0 weights were ever accessible for benchmarking.

### 4.3 Third-Party Derivatives Checked
1. **`nakas/gpttrace-audio-pellav2-onnx`**:
   - Contains `onnx/model_fp32.onnx.data` ($2,321,971,200$ bytes).
   - This is an ONNX export of the 2.32 GB W2V-BERT 2.0 model, not XLS-R.
2. **`varunpadavala/cfic-pellav2-k12-int8`**:
   - Downloaded and inspected checkpoint `293mb/pellav2_k12_int8_final.pt` ($292,876,453$ bytes).
   - State-dict keys: `backbone.encoder.layers.0.ffn1_layer_norm.weight`, `layer_weights: torch.Size([13])`.
   - This is an INT8-quantized 12-layer truncation of W2V-BERT 2.0, not XLS-R 300M.

---

## 5. Checkpoint Verification Verdict

1. **Do we have a genuine pretrained PellaV2 detector?**  
   **NO.** We have successfully loaded the self-supervised foundational backbone (`wav2vec2-xls-r-300m`), but the trained detector weights for the classification head and layer weights do not exist.
2. **Can the original Revision 1 detector checkpoint be recovered?**  
   **NO.** It is absent from Hugging Face Git history, LFS storage, community mirrors, and independent benchmark repositories.
3. **Validity of Previous Smoke Test Scores**:  
   All previous output probabilities were generated by random Kaiming initialization and uniform layer weighting. They have **zero anti-spoofing detection capability**.

---

## 6. Constraints Adhered To

- No replacement head has been trained.
- No architecture modifications have been introduced.
- No ONNX conversion or mobile integration will occur until explicit user direction is provided.
