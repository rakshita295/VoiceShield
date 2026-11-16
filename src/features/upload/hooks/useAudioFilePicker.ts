import { createAudioPlayer } from 'expo-audio';
import * as DocumentPicker from 'expo-document-picker';
import { useCallback, useState } from 'react';

import {
  MAX_DURATION_SECONDS,
  MAX_FILE_SIZE_BYTES,
  MIN_DURATION_SECONDS,
  SUPPORTED_AUDIO_FORMATS,
  UploadedAudioMeta,
  UploadValidationError,
} from '@/features/upload/types/uploadTypes';

export function useAudioFilePicker() {
  const [selectedAudio, setSelectedAudio] = useState<UploadedAudioMeta | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [validationError, setValidationError] = useState<UploadValidationError | null>(null);

  // Helper to extract file extension
  const getFileExtension = (filename: string): string => {
    const idx = filename.lastIndexOf('.');
    return idx !== -1 ? filename.substring(idx).toLowerCase() : '';
  };

  // Inspect and verify audio decoding via native audio player
  const verifyAudioPlayback = async (uri: string): Promise<{ duration: number; isDecodable: boolean }> => {
    try {
      const player = createAudioPlayer({ uri });
      const startTime = Date.now();

      // Poll up to 1200ms for native decoder to read headers and duration
      while (!player.isLoaded && Date.now() - startTime < 1200) {
        await new Promise((resolve) => setTimeout(resolve, 50));
      }

      const duration = player.duration > 0 ? player.duration : 0;
      try {
        player.pause();
      } catch {
        // Safe disposal
      }

      return {
        duration,
        isDecodable: true,
      };
    } catch (err) {
      return {
        duration: 0,
        isDecodable: false,
      };
    }
  };

  const pickAudioFile = useCallback(async (): Promise<UploadedAudioMeta | null> => {
    try {
      setValidationError(null);
      setIsValidating(true);

      const result = await DocumentPicker.getDocumentAsync({
        type: ['audio/*'],
        copyToCacheDirectory: true,
        multiple: false,
      });

      if (result.canceled || !result.assets || result.assets.length === 0) {
        setIsValidating(false);
        return null;
      }

      const asset = result.assets[0];
      const filename = asset.name || 'unnamed_audio';
      const extension = getFileExtension(filename);
      const sizeBytes = asset.size ?? 0;

      // 1. Check for empty file
      if (sizeBytes === 0) {
        const error: UploadValidationError = {
          code: 'FILE_EMPTY',
          title: 'Empty Audio File',
          message: 'The selected file is empty (0 bytes). Please choose a valid voice recording.',
        };
        setValidationError(error);
        setIsValidating(false);
        return null;
      }

      // 2. Check maximum size threshold (25 MB)
      if (sizeBytes > MAX_FILE_SIZE_BYTES) {
        const error: UploadValidationError = {
          code: 'FILE_TOO_LARGE',
          title: 'File Too Large',
          message: `The selected file (${(sizeBytes / (1024 * 1024)).toFixed(1)} MB) exceeds the maximum allowed upload size of 25 MB.`,
        };
        setValidationError(error);
        setIsValidating(false);
        return null;
      }

      // 3. Validate supported audio format
      const isKnownExtension = SUPPORTED_AUDIO_FORMATS.some(
        (fmt) => fmt.extension === extension
      );
      const isAudioMime = asset.mimeType?.startsWith('audio/') ?? false;

      if (!isKnownExtension && !isAudioMime) {
        const error: UploadValidationError = {
          code: 'UNSUPPORTED_FORMAT',
          title: 'Unsupported Audio Format',
          message: `The file format "${extension || 'unknown'}" is not supported. Supported audio formats: WAV, MP3, M4A, AAC, FLAC, and OGG.`,
        };
        setValidationError(error);
        setIsValidating(false);
        return null;
      }

      // 4. Verify actual audio decoder compatibility and duration
      const { duration, isDecodable } = await verifyAudioPlayback(asset.uri);

      if (!isDecodable) {
        const error: UploadValidationError = {
          code: 'CORRUPTED_OR_UNREADABLE',
          title: 'Corrupted Audio Container',
          message: 'The audio stream could not be decoded by the native audio engine. The file may be damaged or using an unsupported codec.',
        };
        setValidationError(error);
        setIsValidating(false);
        return null;
      }

      // 5. Validate duration limits (if decoded duration is non-zero)
      if (duration > 0 && duration < MIN_DURATION_SECONDS) {
        const error: UploadValidationError = {
          code: 'DURATION_TOO_SHORT',
          title: 'Audio Too Short',
          message: `The selected audio is only ${duration.toFixed(1)}s long. VoiceShield requires at least 1.0 second of speech for reliable spectral analysis.`,
        };
        setValidationError(error);
        setIsValidating(false);
        return null;
      }

      if (duration > MAX_DURATION_SECONDS) {
        const error: UploadValidationError = {
          code: 'DURATION_TOO_LONG',
          title: 'Audio Too Long',
          message: `The selected audio (${Math.round(duration / 60)} minutes) exceeds the 5-minute maximum window for on-device voice analysis.`,
        };
        setValidationError(error);
        setIsValidating(false);
        return null;
      }

      const meta: UploadedAudioMeta = {
        uri: asset.uri,
        fileName: filename,
        fileSizeBytes: sizeBytes,
        mimeType: asset.mimeType,
        extension: extension || '.audio',
        durationSec: duration,
        importedAt: new Date(),
        isDecoderVerified: isDecodable,
      };

      setSelectedAudio(meta);
      setIsValidating(false);
      return meta;
    } catch (err: any) {
      const error: UploadValidationError = {
        code: 'PICKER_ERROR',
        title: 'File Selection Failed',
        message: err?.message || 'An unexpected error occurred while picking the audio file.',
      };
      setValidationError(error);
      setIsValidating(false);
      return null;
    }
  }, []);

  const removeAudioFile = useCallback(() => {
    setSelectedAudio(null);
    setValidationError(null);
  }, []);

  return {
    selectedAudio,
    isValidating,
    validationError,
    pickAudioFile,
    removeAudioFile,
    clearError: () => setValidationError(null),
  };
}

