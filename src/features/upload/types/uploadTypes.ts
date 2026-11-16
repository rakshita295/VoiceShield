export interface UploadedAudioMeta {
  uri: string;
  fileName: string;
  fileSizeBytes: number;
  mimeType?: string;
  extension: string;
  durationSec: number;
  importedAt: Date;
  isDecoderVerified: boolean;
}

export type UploadValidationErrorCode =
  | 'FILE_TOO_LARGE'
  | 'FILE_EMPTY'
  | 'UNSUPPORTED_FORMAT'
  | 'DURATION_TOO_SHORT'
  | 'DURATION_TOO_LONG'
  | 'CORRUPTED_OR_UNREADABLE'
  | 'PICKER_ERROR';

export interface UploadValidationError {
  code: UploadValidationErrorCode;
  title: string;
  message: string;
}

export interface SupportedAudioFormat {
  extension: string;
  label: string;
  mimeType: string;
  description: string;
}

export const SUPPORTED_AUDIO_FORMATS: SupportedAudioFormat[] = [
  { extension: '.wav', label: 'WAV', mimeType: 'audio/wav', description: 'Lossless PCM • ML benchmark standard' },
  { extension: '.mp3', label: 'MP3', mimeType: 'audio/mpeg', description: 'Standard compressed audio' },
  { extension: '.m4a', label: 'M4A', mimeType: 'audio/x-m4a', description: 'High-efficiency AAC voice stream' },
  { extension: '.aac', label: 'AAC', mimeType: 'audio/aac', description: 'Advanced Audio Coding' },
  { extension: '.flac', label: 'FLAC', mimeType: 'audio/flac', description: 'Lossless Free Audio (ASVspoof standard)' },
  { extension: '.ogg', label: 'OGG', mimeType: 'audio/ogg', description: 'Ogg Vorbis voice recording' },
];

export const MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024; // 25 MB max
export const MIN_DURATION_SECONDS = 1.0; // At least 1.0s needed for acoustic analysis
export const MAX_DURATION_SECONDS = 300.0; // 5 minutes max

