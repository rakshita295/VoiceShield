export type RecordingStage = 'idle' | 'recording' | 'paused' | 'preview';

export type MicrophonePermissionStatus = 'checking' | 'granted' | 'not_granted' | 'denied';

export interface AudioRecordingMeta {
  uri: string;
  fileName: string;
  durationMs: number;
  fileSizeBytes: number;
  createdAt: Date;
  sampleRate: number;
  channels: number;
  meteringAverage?: number;
}

export interface AudioRecordingConfig {
  sampleRate: number;
  numberOfChannels: number;
  bitRate: number;
  isMeteringEnabled: boolean;
  formatDescription: string;
}

