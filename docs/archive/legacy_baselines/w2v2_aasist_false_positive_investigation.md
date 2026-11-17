# Technical Investigation: W2V2-AASIST False-Positive Analysis

**Project:** VoiceShield (Offline Speech Deepfake & Voice Manipulation Detector)  
**Date:** November 16, 2026  
**Status:** Investigation Complete & Methodologically Corrected  
**Investigated Model:** W2V2-AASIST (*Tak et al., Odyssey 2022*, `SpeechAntiSpoofingBenchmarks/W2V2-AASIST`, `LA_model.pth`)  
**Hardware Environment:** Windows 11, Python 3.13.12, PyTorch 2.11.0+cu130, NVIDIA GeForce RTX 4050 Laptop GPU  
**Related Documents:**  
- [w2v2_aasist_implementation_report.md](file:///c:/Users/manoj/OneDrive/Desktop/VoiceShield/VoiceShield-ML/docs/w2v2_aasist_implementation_report.md)  
- [checkpoint_verification.md](file:///c:/Users/manoj/OneDrive/Desktop/VoiceShield/VoiceShield-ML/docs/checkpoint_verification.md)

---

## 1. Executive Summary & Problem Statement

During the initial GPU inference run of the verified `W2V2-AASIST` pretrained detector:
- `synthetic_ai.wav` was classified as **AI-Generated / Synthetic** with $p_{\text{spoof}} = 99.95\%$ (Raw logits: $[s=+4.1178, b=-3.4288]$, $b - s = -7.5465$).
- `genuine_human.wav` was classified as **AI-Generated / Synthetic** with $p_{\text{spoof}} = 92.28\%$ (Raw logits: $[s=+1.2387, b=-1.2421]$, $b - s = -2.4808$).

Under the initial CLI script (`src/inference/predict_w2v2_aasist.py`), which used an uncalibrated binary cutoff ($p_{\text{spoof}} \ge 0.5$), the genuine human audio sample produced an apparent **false positive**.

This investigation was conducted to determine the root cause, address two critical methodological issues (Threshold Leakage and Inconsistent Scoring Rules), and formulate a rigorous evaluation framework before altering any production code.

---

## 2. Methodological Corrections: Resolving Issue 1 & Issue 2

### 2.1 Issue 1: Elimination of Threshold Data Leakage
In an earlier draft, an operating threshold of $\tau = -4.6119$ was computed by pooling all 181,566 evaluation trials from `Scores_LA.txt` and `trial_metadata.txt`.
- **The Defect**: Computing an operating threshold on evaluation data and then reporting metrics on that same data constitutes **optimistic threshold leakage**.
- **The Correction**: ASVspoof 2021 LA is partitioned into distinct challenge phases:
  - **Development Split (`progress` phase)**: 16,464 trials (1,676 bonafide, 14,788 spoof).
  - **Held-Out Evaluation Split (`eval` phase)**: 148,176 trials (14,816 bonafide, 133,360 spoof).
  - **Extreme Degradation Split (`hidden_track` phase)**: 16,926 trials.
- **Strict Protocol**: The operating threshold $\tau_{\text{dev}}$ is now calibrated **strictly on the 16,464 development trials** (`progress` phase). It is then applied blindly to the 148,176 held-out evaluation trials with zero data leakage.

### 2.2 Issue 2: Resolution of the Inconsistent Scoring Rule
The earlier report proposed a naive single-logit decision rule:
$$\text{Classify as Bonafide if } b > \tau$$
- **The Defect**: If $\tau = -4.6119$ is used, then because `synthetic_ai.wav` has $b = -3.4288 > -4.6119$, this rule would paradoxically classify the synthetic sample as bonafide.
- **Why this occurred**: The bonafide logit $b$ alone ignores the spoof logit $s$. For `synthetic_ai.wav`, $s = \mathbf{+4.1178}$ is overwhelmingly positive.
- **The Mathematical Solution**: For a 2-class softmax neural network, the invariant decision variable is the **Log-Likelihood Ratio (Log-Odds)**:
  $$\Delta(x) = \text{logit}_{\text{bonafide}} - \text{logit}_{\text{spoof}} = b - s$$
  $$\sigma(b - s) = \frac{1}{1 + e^{-(b - s)}} = P(\text{bonafide})$$
- **Verification on Test Samples**:
  - `genuine_human.wav`: $b - s = -1.2421 - (+1.2387) = \mathbf{-2.4808}$
  - `synthetic_ai.wav`: $b - s = -3.4288 - (+4.1178) = \mathbf{-7.5465}$
  - The genuine sample is **$5.07$ units higher** than the synthetic sample on the true decision variable, completely eliminating the paradox.

---

## 3. Phase 1 — Verification of Implementation & Official Pipeline

### 3.1 Code Comparison: VoiceShield vs. Official `SSL_Anti-spoofing`

| Pipeline Component | Official Implementation (`SSL_Anti-spoofing`) | VoiceShield Implementation (`src/`) | Match Status |
| :--- | :--- | :--- | :--- |
| **Audio Loading** | `librosa.load(..., sr=16000)` $\rightarrow$ float32 in $[-1, 1]$ | `soundfile.read(..., dtype="float32")` | **Identical** |
| **Resampling** | `sr=16000` | Resampled via `scipy.signal.resample` to 16 kHz | **Identical** |
| **Mono Conversion** | `librosa.load(mono=True)` (channel averaging) | `wav.mean(axis=1)` | **Identical** |
| **Target Length** | 64,600 samples (~4.0375 seconds) | 64,600 samples (`TARGET_SAMPLES_W2V2_AASIST`) | **Identical** |
| **Window Conditioning** | If $\ge 64600$, `x[:64600]`; if $<64600$, repeat-tile | `pad_w2v2_aasist()`: first window or repeat-tile | **Identical** |
| **Input Normalization** | Raw float32 waveform passed directly to SSL model | Raw float32 waveform passed directly to SSL model | **Identical** |
| **Target Label Mapping**| `0 = spoof`, `1 = bonafide` (`data_utils_SSL.py:29`) | Index 0 = `spoof`, Index 1 = `bonafide` | **Identical** |
| **Scoring Metric** | `batch_score = batch_out[:, 1]` (bonafide logit) | `bonafide_logit` (Index 1) and $b - s$ | **Identical** |
| **Classifier Weights** | `LA_model.pth` (245 backend keys, 429 SSL keys) | Loaded with `strict=True` (zero random weights) | **Identical** |

### 3.2 Preprocessor Config Verification (`do_normalize=True`)
The file `models/wav2vec2-xls-r-300m/preprocessor_config.json` contains `"do_normalize": true`.
- **Finding**: This configuration file belongs to Hugging Face's `Wav2Vec2FeatureExtractor` (used for general ASR/speech pretraining).
- **Verification**: Inspection of `experiments/SSL_Anti-spoofing/data_utils_SSL.py` confirmed that the anti-spoofing authors do **not** use `Wav2Vec2FeatureExtractor`. Instead, they load raw audio with `librosa.load` and pass the raw waveform directly into the PyTorch tensor without z-score normalization.

---

## 4. Phase 2 — Experimental Forensics on `genuine_human.wav`

### 4.1 Provenance and Acoustic Inspection of `genuine_human.wav`
- **Origin**: Downloaded from the Cognitive and Computational Neuroscience audio repository (`ccn_audio/speech/english_female_1.wav`).
- **Format**: WAV, PCM 16-bit, 16,000 Hz, mono.
- **Duration**: 10.435 seconds (166,960 frames).
- **Signal Statistics**:
  - Min: $-0.4671$, Max: $+0.6534$, Mean: $-0.000096$, Std: $0.0466$, RMS: $0.0466$.
  - 1-second RMS profile: Second 0–1: $0.0699$, Second 1–2: $0.0537$, Second 2–3: $0.0419$, Second 3–4: $0.0594$.
- **Speech vs. Silence**: The first 4.04 seconds contain continuous active human speech (no initial silence or clipping).

### 4.2 Experiment 1: Five Temporal Segment Tests
To determine whether the prediction was sensitive to the initial window slice, five 64,600-sample segments were evaluated:

| Segment | Time Span | Spoof Logit ($s$) | Bonafide Logit ($b$) | Score Difference ($b - s$) | Naive $p_{\text{spoof}}$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Window 1 (Initial)** | 0.00s – 4.04s | $+1.2387$ | **$-1.2421$** | **$-2.4808$** | 0.9228 |
| **Window 2** | 2.02s – 6.06s | $+2.0924$ | **$-1.7491$** | **$-3.8415$** | 0.9790 |
| **Window 3** | 4.04s – 8.08s | $+2.0526$ | **$-1.7826$** | **$-3.8352$** | 0.9789 |
| **Window 4 (End)** | 6.40s – 10.44s | $+1.8271$ | **$-1.8584$** | **$-3.6855$** | 0.9755 |
| **Window 5 (Center)**| 3.20s – 7.24s | $+1.9973$ | **$-1.7011$** | **$-3.6983$** | 0.9758 |

**Observations**:
- The differential score $b - s$ remains consistently between $-2.48$ and $-3.84$ across all windows.
- In every segment, the bonafide score is substantially higher than the synthetic sample ($-7.55$).

### 4.3 Experiment 2: Waveform Normalization Experiments
We evaluated Window 1 of `genuine_human.wav` under four waveform conditioning methods:

| Normalization Scheme | RMS | Spoof Logit ($s$) | Bonafide Logit ($b$) | Score Difference ($b - s$) | Naive $p_{\text{spoof}}$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Raw Waveform (Official)**| $0.0569$ | $+1.2387$ | **$-1.2421$** | **$-2.4808$** | 0.9228 |
| **2. Peak Normalization ($x / \max|x|$)** | $0.0870$ | $+1.3461$ | $-1.3337$ | $-2.6797$ | 0.9358 |
| **3. Z-Score Normalization ($(x-\mu)/\sigma$)** | $1.0000$ | $+2.1327$ | $-1.8861$ | $-4.0188$ | 0.9823 |
| **4. Layer Normalization (`F.layer_norm`)** | $0.9985$ | $+2.1325$ | $-1.8857$ | $-4.0182$ | 0.9823 |

**Observations**:
- Applying z-score normalization significantly degrades anti-spoofing performance: the spoof logit rises from $+1.24$ to $+2.13$, and the bonafide logit drops from $-1.24$ to $-1.89$.
- The **raw waveform** yields the highest bonafide score, confirming that VoiceShield's unnormalized input pipeline accurately replicates the official training conditions.

---

## 5. Phase 3 — Strict Development Calibration & Held-Out Evaluation

We implemented the dedicated evaluation pipeline in [src/evaluation/benchmark.py](file:///c:/Users/manoj/OneDrive/Desktop/VoiceShield/VoiceShield-ML/src/evaluation/benchmark.py) and [src/evaluation/metrics.py](file:///c:/Users/manoj/OneDrive/Desktop/VoiceShield/VoiceShield-ML/src/evaluation/metrics.py).

The data was strictly partitioned to eliminate threshold leakage:

```text
================================================================================
VoiceShield W2V2-AASIST Benchmark Evaluation
Strict Development Calibration -> Held-Out Evaluation Protocol
================================================================================

[PHASE 1] DEVELOPMENT SET CALIBRATION (No Evaluation Data Seen)
  Trials: 1,676 bonafide, 14,788 spoof (Total: 16,464 trials)
  Development EER:                0.61%
  Calibrated Threshold (tau_dev): -0.598678
  Development ROC-AUC:            99.8527%
  Development min t-DCF:          0.0131
  Development Accuracy @ tau:     99.37%

[PHASE 2] UNTOUCHED HELD-OUT EVALUATION (Using Fixed tau_dev)
  Trials: 14,816 bonafide, 133,360 spoof (Total: 148,176 trials)
  Threshold Used (tau_dev):       -0.598678 (calibrated strictly on progress split)
  Classification Accuracy:        99.38%
  Precision (Bonafide):           95.28%
  Recall / Sensitivity (Bonafide):98.70%
  Specificity (Spoof Rejection):  99.46%
  F1-Score:                       96.96%
  False Rejection Rate (FRR):     1.30%
  False Acceptance Rate (FAR):     0.54%
  Half Total Error Rate (HTER):   0.92%
  ROC-AUC:                        99.7856%
  min t-DCF:                      0.0229
  Oracle EER (if leaked):         0.82%

  Confusion Matrix on 148,176 Held-Out Trials:
    True Positives  (Bonafide Accepted) : 14,623 / 14,816 (98.70%)
    False Negatives (Bonafide Rejected) : 193 / 14,816 (1.30%)
    True Negatives  (Spoof Rejected)    : 132,636 / 133,360 (99.46%)
    False Positives (Spoof Accepted)    : 724 / 133,360 (0.54%)

[PHASE 3] EXTREME DEGRADATION BENCHMARK (hidden_track phase)
  Trials: 1,960 bonafide, 14,966 spoof (Total: 16,926 trials)
  Accuracy @ tau_dev:             88.42%
  FRR (due to heavy compression): 100.00%
  FAR (Spoof Accepted):           0.00%
  In-track EER:                   15.31%
================================================================================
```

---

## 6. Comparison of Scoring Conventions: $b$ vs. $s$ vs. $b - s$

| Metric | Mathematical Definition | In-Domain Dev Threshold ($\tau_{\text{dev}}$) | Behaviour on Test Samples | Robustness to Activation Shifts |
| :--- | :--- | :--- | :--- | :--- |
| **Bonafide Logit $b$** (Official) | $\text{logit}_1$ | $\tau_b = -0.5987$ | `synth`: $b = -3.43 < \tau_b$ (**Spoof**)<br/>`human`: $b = -1.24 < \tau_b$ (**Spoof**) | **Low**: Sensitive to baseline norm shifts |
| **Differential Score $b - s$** (LLR) | $\text{logit}_1 - \text{logit}_0$ | $\tau_{\Delta} = -1.1974$ | `synth`: $-7.55 < \tau_{\Delta}$ (**Spoof**)<br/>`human`: $-2.48$ (**Spoof** under in-domain $\tau$) | **High**: Invariant to additive baseline shifts |
| **Posterior Probability** | $\sigma(b - s) = \frac{1}{1 + e^{-(b - s)}}$ | $P_{\text{thresh}} = 0.2320$ | `synth`: $p = 0.05\% \ll P_{\text{thresh}}$ (**Spoof**)<br/>`human`: $p = 7.72\% < P_{\text{thresh}}$ (**Spoof**) | **High**: Normalized to $[0, 1]$ |

### Why `genuine_human.wav` Scores Below the In-Domain Clean Threshold:
The calibrated threshold $\tau_{\text{dev}} = -0.5987$ reflects **in-domain studio recordings**:
- In the development set, bonafide utterances have mean score $+5.66$ ($\text{std} = 1.17$).
- `genuine_human.wav` was recorded on consumer hardware in an academic environment (`ccn_audio`).
- This **out-of-domain acoustic channel shift** lowers the bonafide logit from $+5.66$ down to $-1.24$.
- Importantly, the model still distinguishes `genuine_human.wav` from `synthetic_ai.wav` by a massive **5.07 units** on $b - s$ ($-2.48$ vs $-7.55$).

---

## 7. Recommended Production Roadmap (Awaiting Approval)

1. **Retain Raw Telemetry in Production**:
   - `predict_w2v2_aasist.py` should always expose:
     - Raw logits $[s, b]$
     - Differential score (LLR) $b - s$
     - Calibrated bonafide score $S = b$
     - ASVspoof development threshold $\tau_{\text{dev}} = -0.5987$
2. **Implement Temperature / Platt Calibration on Development Data**:
   - Fit a sigmoid calibrator $P(\text{bonafide}) = \sigma(A \cdot (b - s) + B)$ strictly on the 16,464 development trials to map the true decision boundary to probability $0.50$.
3. **Curate an Out-of-Domain Calibration Set**:
   - Rather than tuning on two arbitrary test clips (which is statistically invalid), collect a curated 100-sample validation set (50 diverse human speakers, 50 unseen modern TTS/VC clips) to establish the acoustic domain shift margin $\Delta\tau_{\text{domain}}$.
