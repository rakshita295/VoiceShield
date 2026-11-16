import {
  AudioQuality,
  IOSOutputFormat,
  RecordingOptions,
  RecordingPresets,
} from 'expo-audio';

import { AudioRecordingConfig } from '@/features/recording/types/recordingTypes';

/**
 * ============================================================================
 * VOICESHIELD AUDIO RECORDING CONFIGURATION
 * ============================================================================
 * NOTE FOR FUTURE AI MODEL INTEGRATION (PHASE 5):
 * Different voice deepfake models (e.g. RawNet2, AASIST, Wav2Vec2-XLSR, Whisper)
 * have specific audio frontend input requirements:
 *   - RawNet2 / AASIST typically expect 16,000 Hz, 1-channel (mono), 16-bit PCM.
 *   - General acoustic capture defaults to 44,100 Hz AAC / M4A for high fidelity.
 * 
 * Update these constants when the ML model specification is finalized.
 * All audio recording in VoiceShield reads dynamically from this configuration.
 */
export const DEFAULT_AUDIO_CONFIG: AudioRecordingConfig = {
  sampleRate: 44100,
  numberOfChannels: 1, // Mono preferred for voice speech analysis
  bitRate: 128000,
  isMeteringEnabled: true,
  formatDescription: 'AAC Mono • 44.1 kHz • 128 kbps (Local On-Device)',
};

/**
 * Generate Expo-compatible RecordingOptions based on the active config.
 */
export function getRecordingOptions(
  customConfig?: Partial<AudioRecordingConfig>
): RecordingOptions {
  const config = { ...DEFAULT_AUDIO_CONFIG, ...customConfig };

  return {
    ...RecordingPresets.HIGH_QUALITY,
    extension: '.m4a',
    sampleRate: config.sampleRate,
    numberOfChannels: config.numberOfChannels,
    bitRate: config.bitRate,
    isMeteringEnabled: config.isMeteringEnabled,
    android: {
      extension: '.m4a',
      outputFormat: 'mpeg4',
      audioEncoder: 'aac',
      sampleRate: config.sampleRate,
    },
    ios: {
      extension: '.m4a',
      outputFormat: IOSOutputFormat.MPEG4AAC,
      audioQuality: AudioQuality.HIGH,
      sampleRate: config.sampleRate,
      linearPCMBitDepth: 16,
      linearPCMIsBigEndian: false,
      linearPCMIsFloat: false,
    },
  };
}

