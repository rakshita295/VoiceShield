export type VoiceShieldPrediction = 'GENUINE_HUMAN' | 'AI_GENERATED';

export interface VoiceShieldLatencyBreakdown {
  audioDecodingMs: number;
  preprocessingMs: number;
  inferenceMs: number;
  totalMs: number;
}

export interface VoiceShieldAnalysisResult {
  /** True if the audio is classified as bonafide human speech */
  isGenuine: boolean;
  /** Categorical classification label */
  prediction: VoiceShieldPrediction;
  /** Raw differential model score: s = logit_bonafide - logit_spoof */
  differentialScore: number;
  /** Fixed operating decision threshold (-2.417969) */
  operatingThreshold: number;
  /** Margin above (+) or below (-) the operating threshold */
  scoreMargin: number;
  /** Raw model output logit for bonafide class */
  bonafideLogit: number;
  /** Raw model output logit for spoof class */
  spoofLogit: number;
  /** Number of STFT temporal frames processed */
  numFrames: number;
  /** Duration of analyzed audio segment in seconds (capped at 4.0s) */
  analyzedDurationSec: number;
  /** Sample rate of input audio before internal conditioning */
  originalSampleRate: number;
  /** Detailed on-device execution latencies */
  latency: VoiceShieldLatencyBreakdown;
}
