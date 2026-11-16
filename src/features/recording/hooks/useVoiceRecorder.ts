import {
  getRecordingPermissionsAsync,
  requestRecordingPermissionsAsync,
  setAudioModeAsync,
  useAudioRecorder,
  useAudioRecorderState,
} from 'expo-audio';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { getRecordingOptions, DEFAULT_AUDIO_CONFIG } from '@/features/recording/data/recordingConfig';
import {
  AudioRecordingMeta,
  MicrophonePermissionStatus,
  RecordingStage,
} from '@/features/recording/types/recordingTypes';

export function useVoiceRecorder() {
  const [stage, setStage] = useState<RecordingStage>('idle');
  const [permissionStatus, setPermissionStatus] = useState<MicrophonePermissionStatus>('checking');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [lastRecording, setLastRecording] = useState<AudioRecordingMeta | null>(null);

  // Manual timer backup to ensure smooth 100ms UI clock ticks during recording
  const [elapsedMs, setElapsedMs] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);
  const accumulatedMsRef = useRef<number>(0);

  // Configure recording options with metering enabled
  const recordingOptions = useMemo(() => getRecordingOptions(), []);
  const audioRecorder = useAudioRecorder(recordingOptions);
  // Poll state every 80ms for responsive audio level metering and status
  const recordingState = useAudioRecorderState(audioRecorder, 80);

  // Check microphone permissions on mount
  useEffect(() => {
    let isMounted = true;
    async function checkInitialPermission() {
      try {
        const response = await getRecordingPermissionsAsync();
        if (isMounted) {
          if (response.granted) {
            setPermissionStatus('granted');
          } else if (response.status === 'denied' && !response.canAskAgain) {
            setPermissionStatus('denied');
          } else {
            setPermissionStatus('not_granted');
          }
        }
      } catch (err) {
        if (isMounted) {
          setPermissionStatus('not_granted');
        }
      }
    }

    checkInitialPermission();
    return () => {
      isMounted = false;
    };
  }, []);

  // Request microphone permission on user interaction
  const requestPermission = useCallback(async (): Promise<boolean> => {
    try {
      setErrorMessage(null);
      const response = await requestRecordingPermissionsAsync();
      if (response.granted) {
        setPermissionStatus('granted');
        return true;
      } else if (response.status === 'denied' && !response.canAskAgain) {
        setPermissionStatus('denied');
        return false;
      } else {
        setPermissionStatus('not_granted');
        return false;
      }
    } catch (err) {
      setErrorMessage('Failed to request microphone permission. Please check system settings.');
      return false;
    }
  }, []);

  // Timer helpers
  const startTimer = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    startTimeRef.current = Date.now();
    timerRef.current = setInterval(() => {
      const currentDelta = Date.now() - startTimeRef.current;
      setElapsedMs(accumulatedMsRef.current + currentDelta);
    }, 100);
  }, []);

  const pauseTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    accumulatedMsRef.current += Date.now() - startTimeRef.current;
    setElapsedMs(accumulatedMsRef.current);
  }, []);

  const resetTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    startTimeRef.current = 0;
    accumulatedMsRef.current = 0;
    setElapsedMs(0);
  }, []);

  // Start recording
  const startRecording = useCallback(async () => {
    try {
      setErrorMessage(null);

      // Verify permission
      let hasPermission = permissionStatus === 'granted';
      if (!hasPermission) {
        hasPermission = await requestPermission();
        if (!hasPermission) {
          setErrorMessage('Microphone access is required to capture voice audio.');
          return;
        }
      }

      // Configure global audio session for recording
      await setAudioModeAsync({
        allowsRecording: true,
        playsInSilentMode: true,
      });

      // Prepare and begin recording
      await audioRecorder.prepareToRecordAsync();
      audioRecorder.record();

      resetTimer();
      startTimer();
      setStage('recording');
    } catch (err: any) {
      setErrorMessage(err?.message ?? 'Failed to start audio recording.');
      setStage('idle');
      resetTimer();
    }
  }, [permissionStatus, requestPermission, audioRecorder, resetTimer, startTimer]);

  // Pause recording
  const pauseRecording = useCallback(() => {
    try {
      if (stage !== 'recording') return;
      audioRecorder.pause();
      pauseTimer();
      setStage('paused');
    } catch (err: any) {
      setErrorMessage(err?.message ?? 'Failed to pause recording.');
    }
  }, [stage, audioRecorder, pauseTimer]);

  // Resume recording
  const resumeRecording = useCallback(() => {
    try {
      if (stage !== 'paused') return;
      audioRecorder.record();
      startTimer();
      setStage('recording');
    } catch (err: any) {
      setErrorMessage(err?.message ?? 'Failed to resume recording.');
    }
  }, [stage, audioRecorder, startTimer]);

  // Stop recording
  const stopRecording = useCallback(async () => {
    try {
      if (stage !== 'recording' && stage !== 'paused') return;

      pauseTimer();
      await audioRecorder.stop();

      const finalDuration = Math.max(elapsedMs, recordingState.durationMillis);
      const uri = audioRecorder.uri;

      if (!uri) {
        throw new Error('Recorded audio file URI was not generated.');
      }

      const timestamp = new Date();
      const meta: AudioRecordingMeta = {
        uri,
        fileName: `voiceshield_${timestamp.getTime()}.m4a`,
        durationMs: finalDuration,
        fileSizeBytes: recordingState.fileSize,
        createdAt: timestamp,
        sampleRate: DEFAULT_AUDIO_CONFIG.sampleRate,
        channels: DEFAULT_AUDIO_CONFIG.numberOfChannels,
      };

      setLastRecording(meta);
      setStage('preview');
    } catch (err: any) {
      setErrorMessage(err?.message ?? 'Failed to stop audio recording.');
      setStage('idle');
      resetTimer();
    }
  }, [stage, pauseTimer, audioRecorder, elapsedMs, recordingState, resetTimer]);

  // Discard and reset back to idle
  const discardRecording = useCallback(() => {
    resetTimer();
    setLastRecording(null);
    setErrorMessage(null);
    setStage('idle');
  }, [resetTimer]);

  // Watch for system interruptions
  useEffect(() => {
    if (recordingState.mediaServicesDidReset && (stage === 'recording' || stage === 'paused')) {
      setErrorMessage('Media services were reset by the system. Recording ended.');
      setStage('idle');
      resetTimer();
    }
  }, [recordingState.mediaServicesDidReset, stage, resetTimer]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  // Compute normalized audio metering for real-time waveform visualization (0.0 to 1.0)
  // Decibels typically range from -60 (silence) to 0 (peak) on Android
  const rawMetering = recordingState.metering;
  const isMeteringAvailable = rawMetering !== undefined && rawMetering !== null;
  const normalizedLevel = useMemo(() => {
    if (!isMeteringAvailable || typeof rawMetering !== 'number') {
      return 0.15; // Ambient baseline floor
    }
    // Map -60 dB .. 0 dB to 0.0 .. 1.0
    const clamped = Math.min(0, Math.max(-60, rawMetering));
    return Math.max(0.12, (clamped + 60) / 60);
  }, [rawMetering, isMeteringAvailable]);

  return {
    stage,
    permissionStatus,
    errorMessage,
    lastRecording,
    elapsedMs,
    rawMetering,
    isMeteringAvailable,
    normalizedLevel,
    isRecording: recordingState.isRecording,
    fileSize: recordingState.fileSize,
    requestPermission,
    startRecording,
    pauseRecording,
    resumeRecording,
    stopRecording,
    discardRecording,
    clearError: () => setErrorMessage(null),
  };
}

