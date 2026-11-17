# VoiceShield — Mobile ML Integration Plan
**Document Version:** 1.0  
**Target Mobile Application:** [rakshita295/VoiceShield](https://github.com/rakshita295/VoiceShield)  
**Selected ML Model:** VoiceShield LFCC-CMVN-CNN-BiLSTM-Attention (V2, Checkpoint Epoch 3)  
**Export Artifact:** `models/voiceshield_v2.onnx` (6.95 MB, 1,736,706 parameters)  
**Decision Threshold:** $\tau = -2.417969$

---

## 1. Executive Summary

This document specifies the end-to-end integration strategy for deploying the custom VoiceShield deepfake speech detection model directly onto Rakshita's existing React Native / Expo mobile application.

The target model is a completely custom, lightweight neural network (~1.74M parameters, 6.95 MB FP32 ONNX) trained from scratch on ASVspoof 2019 LA that operates without heavy pretrained audio backbones (such as 300M+ parameter XLS-R or Wav2Vec2). The model achieves an official Evaluation EER of **8.736%** and ROC-AUC of **0.96327**, executing in **under 15 ms** on modern mobile hardware.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Mobile Audio Pipeline Architecture                 │
└─────────────────────────────────────────────────────────────────────────┘
   [Audio Input]  (Microphone via expo-audio or File via expo-document-picker)
         │
         ▼
   [Decoding & Resampling] (16,000 Hz, 16-bit Mono PCM, normalized to [-1.0, 1.0])
         │
         ▼
   [LFCC Front-End] (20 static + 20 delta + 20 delta-delta = 60 channels)
         │
         ▼
   [Utterance-Level CMVN] (Per-channel zero mean, unit variance, eps=1e-6)
         │  Tensor shape: [1, 1, 60, T_frames]
         ▼
   [ONNX Runtime Mobile] (voiceshield_v2.onnx — CNN-BiLSTM-Attention Backbone)
         │  Logits: [1, 2] | Differential score: logit_1 - logit_0
         ▼
   [Classification Engine] (Score >= -2.417969 ? Genuine Human : AI Synthetic)
         │
         ▼
   [React Native UI] (Interactive Result Card & History Logging)
```

---

## 2. Rakshita's Mobile Application Inspection Findings

An audit of Rakshita's repository (`rakshita295/VoiceShield`) revealed the following architecture:

### 2.1 Framework & Core Dependencies
- **Framework:** React Native `0.88.1` with React `19.3.0`
- **Expo SDK:** Expo `~58.0.25` with Expo Router `~58.0.31`
- **Language:** TypeScript `~6.0.3`
- **Workflow:** Expo Managed Workflow with Continuous Native Generation (CNG / Prebuild)
- **Audio Recording:** `expo-audio` `~58.0.12`
- **File Selection:** `expo-document-picker` `~58.0.5`
- **Animations & UI:** `react-native-reanimated` `4.7.1`, `react-native-svg` `15.16.0`, `expo-linear-gradient` `~58.0.4`

### 2.2 Application Structure & Integration Hooks
- **Navigation:** File-based routing via `src/app/`:
  - `src/app/index.tsx`: Dashboard / Home Screen
  - `src/app/record.tsx`: Audio Recording Coordinator (`src/features/recording/`)
  - `src/app/upload.tsx`: File Upload Coordinator (`src/features/upload/`)
  - `src/app/history.tsx`: Scan History Screen
  - `src/app/insights.tsx`: Security Insights & Detection Analytics
  - `src/app/settings.tsx`: App Settings & Threshold Configuration
- **Existing Integration Points (Placeholders):**
  - In `src/features/recording/components/AudioPreviewScreen.tsx`: The primary "Analyze Recording" button triggers an informational modal stating:
    > *"The audio recording is safely stored locally. In Phase 5, this audio sample will be fed directly into our on-device ONNX deepfake detection engine to compute synthetic voice probability, spectral anomalies, and acoustic authenticity scores."*
  - In `src/features/upload/components/UploadedAudioPreviewScreen.tsx`: Identical "Analyze File" action with placeholder modal.
- **Ready for Hook-in:** Both preview screens are already structured to receive a classification result object and display results or route to a dedicated results modal/view.

---

## 3. Audio & Preprocessing Specifications

The trained model does **not** accept raw PCM audio directly; it requires exact feature extraction mirroring the Python pipeline.

### 3.1 Audio Input Requirements
| Parameter | Required Value | Notes |
| :--- | :--- | :--- |
| **Sampling Rate** | `16,000 Hz` | Downsample if recorded at 44.1 kHz or 48 kHz |
| **Channels** | `1` (Mono) | If stereo, average left + right channels |
| **Sample Format** | Float32 (`[-1.0, 1.0]`) | Convert from 16-bit signed integer PCM (`sample / 32768.0`) |
| **Minimum Duration**| `1.0 second` (16,000 samples) | Model handles dynamic lengths; ~3-5 seconds is ideal |

### 3.2 LFCC Front-End Parameters
| Parameter | Value | Details |
| :--- | :--- | :--- |
| **FFT Size ($N_{fft}$)** | `512` | 512-point Short-Time Fourier Transform |
| **Window Length** | `400` samples | 25.0 ms analysis window |
| **Hop Length** | `160` samples | 10.0 ms frame shift (100 frames/sec) |
| **Window Function** | Periodic Hann | $w[n] = 0.5 - 0.5 \cos(2\pi n / 400)$ |
| **STFT Centering** | `True` | Reflective padding of 200 samples on both ends |
| **Filterbank Type** | Linear Frequency | 20 linearly spaced triangular filters across $[0, 8000\text{ Hz}]$ |
| **Log Energy** | Natural Logarithm | $\ln(\text{energy} + 10^{-6})$ |
| **Cepstral Coeffs ($N_{lfcc}$)** | `20` | Computed via type-II Discrete Cosine Transform (DCT-II) |
| **Delta Coeffs** | `20` | Velocity with window width $N=2$ |
| **Delta-Delta Coeffs** | `20` | Acceleration with window width $N=2$ |
| **Total Feature Dimension** | `60` | $[static_{0..19}, delta_{0..19}, delta\text{-}delta_{0..19}]$ |

### 3.3 Utterance-Level CMVN Formula
For each of the $C=60$ feature channels across all $T$ temporal frames:
$$\mu_c = \frac{1}{T} \sum_{t=1}^{T} X_{c, t}$$
$$\sigma_c = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (X_{c, t} - \mu_c)^2 + \epsilon}$$
$$X'_{c, t} = \frac{X_{c, t} - \mu_c}{\sigma_c}$$
where $\epsilon = 1.0 \times 10^{-6}$.

---

## 4. Mobile Execution Strategy & Architecture Trade-offs

Extracting 60-channel LFCC features and running a 1.74M-parameter BiLSTM on Android requires choosing an execution architecture that guarantees numerical precision and sub-second latency.

### 4.1 Evaluation of Implementation Approaches

| Approach | Latency (3s audio) | Numerical Parity | Expo Compatibility | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **Approach 1: Pure JavaScript in Hermes** | 2,500 – 6,000 ms | Risky (Floating point differences in JS math) | Full (Expo Go) | **REJECTED**: Severe UI stutter, high GC pressure. |
| **Approach 2: Native Kotlin Expo Module (In-App)** | **15 – 35 ms** | **Exact** (Matches Torchaudio filterbank & ONNX) | Full (Expo Prebuild / Dev Build) | **RECOMMENDED**: Optimal performance & seamless Expo integration. |
| **Approach 3: C++ JNI Native Module** | 10 – 20 ms | Exact | Requires NDK / CMake maintenance | Alternative for extreme optimization. |

### 4.2 Recommended Architecture: Native Kotlin Expo Module
Create an Expo native module: `modules/voiceshield-detector/` using the Expo Modules API (`expo-modules-core`):
1. **Audio Decoding:** Reads recorded audio file (`.wav` or `.m4a`), extracts raw PCM samples at 16 kHz using Android's standard `MediaExtractor` / `MediaCodec` or standard PCM header parser.
2. **Feature Extraction:** Lightweight Kotlin DSP class (`LFCCProcessor.kt`):
   - Computes 512-point FFT (using KissFFT or pure Kotlin Cooley-Tukey).
   - Multiplies with precomputed $20 \times 257$ linear filterbank matrix.
   - Applies DCT-II matrix multiplication.
   - Computes deltas and delta-deltas.
   - Applies utterance-level CMVN.
3. **ONNX Runtime Mobile Execution:**
   - Bundles `com.microsoft.onnxruntime:onnxruntime-android:1.18.0` via Gradle.
   - Executes `voiceshield_v2.onnx` on CPU or NNAPI.
4. **TypeScript Interface:** Exposes a simple Promise-based function:
   ```typescript
   import { detectDeepfake } from 'voiceshield-detector';
   const result = await detectDeepfake(audioUri);
   ```

---

## 5. ONNX Model Specifications

### 5.1 Model File Details
- **File Name:** `voiceshield_v2.onnx`
- **File Size:** `6.95 MB` (6,953,979 bytes)
- **Opset Version:** `14` (Compatible with ONNX Runtime Mobile 1.14+)
- **Parameters:** 1,736,706 (FP32)

### 5.2 Input Tensor
- **Name:** `lfcc_features`
- **Data Type:** `Float32`
- **Shape:** `[1, 1, 60, T_frames]`
  - Dimension 0: Batch size (fixed to 1 on device)
  - Dimension 1: Channel dimension (1)
  - Dimension 2: Cepstral feature bins (60)
  - Dimension 3: Temporal frame count ($T_{frames} = \lfloor L_{samples} / 160 \rfloor + 1$)

### 5.3 Output Tensors
- **Output 1:** `logits`
  - Shape: `[1, 2]`
  - Index 0: `logit_spoof` (AI-generated speech)
  - Index 1: `logit_bonafide` (Genuine human speech)
- **Output 2:** `score`
  - Shape: `[1]`
  - Value: Differential log-likelihood ratio $s = \text{logit}_{\text{bonafide}} - \text{logit}_{\text{spoof}}$

### 5.4 Decision Logic & Threshold Calibration
- **Threshold:** $\tau = -2.417969$ (Selected on ASVspoof 2019 LA development split)
- **Classification Rule:**
  $$\text{Prediction} = \begin{cases} \text{"GENUINE\_HUMAN"}, & \text{if } s \ge -2.417969 \\ \text{"AI\_GENERATED"}, & \text{if } s < -2.417969 \end{cases}$$
- **Authenticity Confidence Calculation:**
  $$\text{Confidence}_{\text{Human}} = \frac{1}{1 + e^{-(s - \tau) / 3.0}} \times 100\%$$

---

## 6. Step-by-Step Implementation Guide for Mobile

### Step 1: Export and Place the ONNX Model
In `VoiceShield-ML`, run:
```powershell
python scripts/export_onnx.py --checkpoint experiments/custom_lfcc_cmvn_v1/best_model.pt --output models/voiceshield_v2.onnx
```
Copy `models/voiceshield_v2.onnx` into the mobile app's assets:
```
VoiceShield-Mobile/
└── assets/
    └── models/
        └── voiceshield_v2.onnx
```

### Step 2: Configure Android Native Dependencies
Add ONNX Runtime to the Android app build via `build.gradle` or Expo Config Plugin:
```groovy
dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.18.0'
}
```

### Step 3: Kotlin Feature Extractor & Detector Module
Below is the reference Kotlin implementation for the native module:

```kotlin
package com.voiceshield.detector

import android.content.Context
import ai.onnxruntime.*
import java.nio.FloatBuffer
import kotlin.math.*

class VoiceShieldDetector(private val context: Context) {
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null
    private val THRESHOLD = -2.417969f

    init {
        // Load ONNX model from Android assets
        val modelBytes = context.assets.open("models/voiceshield_v2.onnx").readBytes()
        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        ortSession = ortEnv.createSession(modelBytes, sessionOptions)
    }

    fun analyzeAudio(pcmFloatSamples: FloatArray): DetectionResult {
        val startTime = System.currentTimeMillis()

        // 1. Extract 60-dim LFCC with utterance CMVN: Shape [1, 1, 60, T]
        val (features, numFrames) = LFCCProcessor.extractFeaturesWithCMVN(pcmFloatSamples)

        // 2. Prepare ONNX FloatBuffer Tensor
        val tensorShape = longArrayOf(1, 1, 60, numFrames.toLong())
        val tensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(features), tensorShape)

        // 3. Run Inference
        val results = ortSession!!.run(mapOf("lfcc_features" to tensor))
        val score = (results[1].value as FloatArray)[0]
        val logits = (results[0].value as Array<FloatArray>)[0]

        val isGenuine = score >= THRESHOLD
        val confidence = 1.0f / (1.0f + exp(-(score - THRESHOLD) / 3.0f))
        val elapsedMs = System.currentTimeMillis() - startTime

        return DetectionResult(
            isGenuine = isGenuine,
            differentialScore = score,
            spoofLogit = logits[0],
            bonafideLogit = logits[1],
            confidencePercent = if (isGenuine) confidence * 100f else (1.0f - confidence) * 100f,
            latencyMs = elapsedMs
        )
    }
}
```

### Step 4: Hooking into Rakshita's UI
Replace the placeholder modal in `src/features/recording/components/AudioPreviewScreen.tsx`:

```typescript
// In AudioPreviewScreen.tsx:
import { VoiceShieldDetector } from '@/modules/voiceshield-detector';

const handleAnalyzePress = async () => {
  try {
    setIsAnalyzing(true);
    const result = await VoiceShieldDetector.analyze(recording.uri);
    
    // Navigate to results or update UI state
    router.push({
      pathname: '/results',
      params: {
        isGenuine: result.isGenuine ? '1' : '0',
        score: result.differentialScore.toFixed(3),
        confidence: result.confidencePercent.toFixed(1),
        latencyMs: result.latencyMs,
      }
    });
  } catch (error) {
    Alert.alert('Analysis Failed', 'Could not process audio. Please try again.');
  } finally {
    setIsAnalyzing(false);
  }
};
```

---

## 7. Numerical Parity Verification Checkpoints

To ensure that the mobile model matches the Python baseline identically, use the sample test recordings located in `datasets/samples/`:
1. `genuine_human.wav`:
   - Expected Python Score: `+14.80`
   - Mobile Verification Target: Score $\ge +14.0$ (GENUINE_HUMAN, >99.9% confidence)
2. `synthetic_ai.wav`:
   - Expected Python Score: `-8.99`
   - Mobile Verification Target: Score $\le -8.0$ (AI_GENERATED, >99.9% confidence)

Any mobile run deviating by more than $\Delta = \pm 0.2$ in score indicates an STFT windowing or CMVN variance normalization discrepancy.
