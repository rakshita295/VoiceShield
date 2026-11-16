# VoiceShield – Mobile Application

> **On-Device AI Voice Deepfake & Synthetic Speech Protection**  
> *React Native & Expo Mobile Client*

---

## 1. Project Overview

**VoiceShield** is a modern, privacy-first Android mobile application designed to protect individuals and organizations from malicious AI voice clones, synthesized speech scams, and deepfake impersonations. 

Built with **React Native** and **Expo SDK 58**, the mobile client provides an intuitive interface for capturing live audio or importing pre-recorded audio files, previewing waveforms, and managing voice authenticity verification entirely on the user's device.

---

## 2. Problem Statement

With the rapid proliferation of advanced generative voice synthesis (TTS) and voice conversion (VC) technologies, audio deepfakes have become indistinguishable from genuine human speech. These technologies are increasingly weaponized for:
- CEO voice impersonation and financial wire fraud.
- Social engineering attacks targeting family members ("virtual kidnapping" scams).
- Spoofing automated voice biometric authentication systems.

Most existing countermeasures rely on cloud-based APIs, introducing latency, dependency on network connectivity, and severe user privacy risks by sending sensitive voice biometrics across the Internet. **VoiceShield** addresses this challenge by designing an on-device countermeasure workflow that prioritizes mobile performance, zero data exfiltration, and cross-platform compatibility.

---

## 3. Main Application Features

### 🛡️ Brand Identity & Splash Screen
- **Code-Generated Vector Branding:** High-resolution geometric shield logo featuring audio waveform bars and AI neural circuit traces.
- **Cinematic React Splash Animation:** 3-second presentation with subtle breathing glow and smooth 500 ms exit transition, fully synchronized with Expo Router without blocking cold-boot navigation.

### 🚀 Interactive Onboarding Flow
- **3-Screen Educational Storyline:**
  1. *Real-time Deepfake Detection:* Explains how neural synthesis artifacts are captured.
  2. *Your Privacy Matters:* Emphasizes 100% on-device processing and zero audio upload to cloud servers.
  3. *Shield Your Voice:* Introduces call and voice note verification.
- **Natural Touch Gestures:** Smooth horizontal swiping powered by `react-native-gesture-handler` and `react-native-reanimated`, integrated alongside Skip, Next, Back, and Get Started navigation.

### 📊 Security Dashboard & Bottom Navigation
- **Security Score & Shield Status:** Real-time visual threat indicator and status badge.
- **Quick Action Center:** Direct access to live microphone recording and audio file importation.
- **4-Tab Navigation Architecture:** Persistent bottom bar featuring:
  - `Home`: Main authenticity overview and quick actions.
  - `History`: Placeholder registry for analyzed voice notes.
  - `Insights`: Synthetic attack telemetry and analytics.
  - `Settings`: Privacy toggles, sensitivity threshold controls, and permissions.

### 🎙️ Audio Recording Studio
- **4-State Coordinator Architecture:**
  - `RecordAudio`: Clean pre-recording preparation screen with microphone permission checks.
  - `ActiveRecording`: Real-time audio metering with responsive, glowing waveform visualizer.
  - `PausedRecording`: Resume, finish, or discard controls.
  - `AudioPreview`: Seekable playback scrubber, elapsed time display, file metadata, and verification triggers.
- **Local Storage Management:** Audio recorded locally in high-fidelity PCM format using Expo's modern `expo-audio` module.

### 📁 Audio Upload & Inspection Engine
- **Native Document Picker:** Integrated via `expo-document-picker` leveraging the Android Storage Access Framework (SAF) with zero dangerous runtime permission requirements.
- **Multi-Format Validation:** Validates container formats including `.wav`, `.mp3`, `.m4a`, `.aac`, `.flac`, and `.ogg`.
- **Integrity Safeguards:** Enforces strict audio bounds (file size $\le 25\text{ MB}$, duration between $1.0\text{s}$ and $300.0\text{s}$) with automated audio decoder verification.
- **Preview & Re-selection:** Full scrubbable playback preview with metadata summary card and one-tap replacement.

### 🔍 Verification Hand-off Modal
- Dedicated confirmation and analysis modal that presents file parameters and prepares the audio payload for machine-learning inference.

---

## 4. Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Framework** | [React Native](https://reactnative.dev/) | `0.88.1` | Native mobile application runtime |
| **Platform** | [Expo SDK](https://expo.dev/) | `58.0.25` | Modern mobile toolchain & Continuous Native Generation |
| **Language** | [TypeScript](https://www.typescriptlang.org/) | `6.0.3` | Type-safe architecture and contracts |
| **Routing** | [Expo Router](https://docs.expo.dev/router/introduction/) | `58.0.31` | Declarative, file-based routing |
| **Animations** | [React Native Reanimated](https://docs.swmansion.com/react-native-reanimated/) | `4.7.1` | 60/120 FPS native thread animations |
| **Gestures** | [React Native Gesture Handler](https://docs.swmansion.com/react-native-gesture-handler/) | `3.3.0` | Native touch and swipe gesture coordination |
| **Audio I/O** | [expo-audio](https://docs.expo.dev/versions/latest/sdk/audio/) | `58.0.12` | Low-latency audio recording and playback |
| **Document I/O** | [expo-document-picker](https://docs.expo.dev/versions/latest/sdk/document-picker/) | `58.0.5` | Android SAF native file selection |
| **Vector Graphics**| [react-native-svg](https://github.com/software-mansion/react-native-svg) | `15.16.0` | Resolution-independent logos and illustrations |
| **Visual Effects** | [expo-linear-gradient](https://docs.expo.dev/versions/latest/sdk/linear-gradient/) | `58.0.4` | Premium cyber-security dark theme gradients |

---

## 5. Project Directory Structure

```text
voiceshield-mobile/
├── assets/                          # Static branding, launcher icons, and splash assets
│   ├── expo.icon/                   # Adaptive icon definitions
│   └── images/                      # App icons and graphics
├── src/
│   ├── app/                         # Expo Router screen routes (File-based routing)
│   │   ├── _layout.tsx              # Root Layout: Theme provider, Splash, Gesture root
│   │   ├── index.tsx                # Entry point: Onboarding gate & Home shell
│   │   ├── history.tsx              # History tab screen
│   │   ├── insights.tsx             # Insights tab screen
│   │   ├── settings.tsx             # Settings tab screen
│   │   ├── onboarding.tsx           # Standalone onboarding route
│   │   ├── record.tsx               # Audio recording studio route
│   │   └── upload.tsx               # Audio file upload & preview route
│   ├── components/                  # Reusable presentation and UI components
│   │   ├── animations/              # FlowingWaveform and glowing effects
│   │   ├── branding/                # VoiceShieldLogo & VoiceShieldSplashScreen
│   │   ├── dashboard/               # Header, authenticity card, quick actions
│   │   ├── navigation/              # VoiceShieldBottomNav tab bar
│   │   └── onboarding/              # Slide headers, footers, pagination dots
│   ├── constants/                   # Design system tokens and brand palette
│   │   └── theme.ts                 # Colors, gradients, typography, spacing
│   ├── features/                    # Domain-driven modular features
│   │   ├── dashboard/               # Shell state, tab coordination, placeholders
│   │   ├── onboarding/              # Illustrations, slide data, completion store
│   │   ├── recording/               # useVoiceRecorder, useVoicePlayer, 4-state UI
│   │   └── upload/                  # useAudioFilePicker, validation, preview
│   └── hooks/                       # Custom theme and platform utilities
├── .gitignore                       # Production repository exclusion rules
├── AGENTS.md                        # Expo SDK 58 agent development guidelines
├── app.json                         # Expo configuration (CNG, permissions, plugins)
├── package.json                     # NPM dependency manifest
└── tsconfig.json                    # TypeScript compiler configuration
```

---

## 6. Installation & Prerequisites

### Prerequisites
- [Node.js](https://nodejs.org/) (v20.x or v22.x LTS recommended)
- [npm](https://www.npmjs.com/) (bundled with Node.js)
- Android device or Android Studio emulator (Android 10+ / API 29+)

### Installation
Clone the repository and install the validated Expo SDK 58 dependencies:

```bash
git clone https://github.com/<your-org>/voiceshield-mobile.git
cd voiceshield-mobile
npm install
```

---

## 7. Running the Application

### 7.1 Start the Metro Development Server
```bash
npx expo start
```
From the interactive terminal menu, press:
- `a` to launch on a connected Android device or running Android emulator.
- `r` to reload the bundle.

### 7.2 Native Development Build (Recommended for Audio & Permissions)
Because `expo-audio` requires native audio recording modules, run the development build:
```bash
npx expo run:android
```
This compiles the native Android project via Continuous Native Generation (CNG) and installs the debug APK directly on your device.

---

## 8. Android Physical Device Testing Instructions

1. **Enable Developer Options on Device:**
   - Go to **Settings > About Phone** and tap **Build Number** 7 times.
   - Go to **Settings > System > Developer Options** and enable **USB Debugging**.
2. **Connect Device via USB:**
   - Verify connection by running:
     ```bash
     adb devices
     ```
3. **Run Development Build:**
   ```bash
   npx expo run:android
   ```
4. **Test Permissions & Audio Features:**
   - **Splash Screen:** Observe 3-second logo presentation, breathing cyan glow, and 500 ms fade-out.
   - **Onboarding:** Swipe through the 3 onboarding cards. Verify smooth horizontal transitions and tap **Get Started**.
   - **Audio Recording:** Tap **Record Audio** on the dashboard. Grant microphone permission when prompted. Record an utterance, observe the live metering waveform, pause, and preview playback.
   - **Audio Upload:** Tap **Upload Audio** on the dashboard. Select an audio file (`.wav`, `.mp3`, `.m4a`, etc.) from your device storage. Verify file size, duration, and audio playback.

---

## 9. Current Implementation Status

> [!IMPORTANT]
> **Machine Learning Engine Integration Notice:**  
> The VoiceShield on-device machine learning detection engine (`VoiceShieldNet`) is currently being developed and trained from scratch by the ML engineering team in a separate, decoupled repository (`voiceshield-ml`).  
> **Deepfake voice detection is NOT yet operational in this mobile client.**  
> The current mobile build includes complete Phase 1–4 capabilities (UI/UX, onboarding, dashboard, native audio recording, file validation, playback preview, and analysis handoff modals). Tapping "Verify Voice Authenticity" currently displays the verification preview modal in preparation for the forthcoming ONNX Runtime engine.

### Implementation Checklist
- [x] **Phase 1 – Branding & Splash:** Custom shield logo, dark-mode purple-cyan palette, 3s animated splash.
- [x] **Phase 2 – Animated Onboarding:** 3 custom vector illustrations, touch swipe gesture support, completion persistence.
- [x] **Phase 3 – Home Dashboard:** 4-tab shell navigation, threat score card, quick actions, empty states.
- [x] **Phase 4A – Audio Recording Studio:** Native microphone capture, metering waveforms, pause/resume, seekable playback.
- [x] **Phase 4B – Audio Upload Workflow:** Storage Access Framework picker, format validation, duration checks, preview player.
- [ ] **Phase 5 – On-Device AI Engine Integration:** Deployment of `voiceshield_net.onnx` via `onnxruntime-react-native` (Pending ML model training).

---

## 10. Future Development Roadmap

1. **On-Device ONNX Inference (Phase 5):**
   - Embed the trained, quantized `voiceshield_net.onnx` model (<2 MB).
   - Implement local 16 kHz audio preprocessing and Log-Mel spectrogram generation directly on the Android device.
2. **Detection Results Dashboard:**
   - Detailed authenticity confidence rating ($0\text{–}100\%$).
   - Spectral artifact breakdown (vocoder phase errors, synthetic harmonic anomalies).
   - Flagging of inconclusive/uncertain results requiring secondary inspection.
3. **Encrypted History Registry:**
   - SQLite / SecureStore local database storing timestamped verification records.
4. **Real-time In-Call Protection:**
   - Accessibility / audio-stream monitoring service alerting users to incoming synthesized voice calls.

---

## 11. License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
