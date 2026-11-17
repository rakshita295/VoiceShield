import { requireNativeModule } from 'expo-modules-core';
import { VoiceShieldAnalysisResult } from './src/VoiceShieldDetector.types';

const VoiceShieldDetector = requireNativeModule('VoiceShieldDetector');

/**
 * Analyzes a local audio file and returns on-device VoiceShield deepfake detection results.
 *
 * @param audioUri Local file URI (e.g. "file:///path/to/audio.wav" or "/path/to/audio.m4a")
 * @returns Complete VoiceShieldAnalysisResult containing prediction, raw score and latency
 */
export async function analyzeAudioFile(audioUri: string): Promise<VoiceShieldAnalysisResult> {
  return await VoiceShieldDetector.analyzeAudioFile(audioUri);
}

/**
 * Returns the fixed operating threshold (-2.417969).
 */
export function getOperatingThreshold(): number {
  return VoiceShieldDetector.getThreshold();
}

export * from './src/VoiceShieldDetector.types';
