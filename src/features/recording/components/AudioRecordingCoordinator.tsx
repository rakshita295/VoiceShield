import { router } from 'expo-router';
import React, { useEffect } from 'react';
import { Alert, BackHandler, StyleSheet, View } from 'react-native';

import { ActiveRecordingScreen } from '@/features/recording/components/ActiveRecordingScreen';
import { AudioPreviewScreen } from '@/features/recording/components/AudioPreviewScreen';
import { PausedRecordingScreen } from '@/features/recording/components/PausedRecordingScreen';
import { RecordAudioScreen } from '@/features/recording/components/RecordAudioScreen';
import { useVoiceRecorder } from '@/features/recording/hooks/useVoiceRecorder';
import { BrandColors } from '@/constants/theme';

export const AudioRecordingCoordinator: React.FC = () => {
  const {
    stage,
    permissionStatus,
    errorMessage,
    lastRecording,
    elapsedMs,
    rawMetering,
    isMeteringAvailable,
    normalizedLevel,
    fileSize,
    requestPermission,
    startRecording,
    pauseRecording,
    resumeRecording,
    stopRecording,
    discardRecording,
  } = useVoiceRecorder();

  // Android hardware back button handler
  useEffect(() => {
    const onBackPress = () => {
      if (stage === 'recording' || stage === 'paused') {
        Alert.alert(
          'Cancel Recording?',
          'Are you sure you want to discard this recording and return to the dashboard?',
          [
            { text: 'Keep Recording', style: 'cancel' },
            {
              text: 'Discard',
              style: 'destructive',
              onPress: () => {
                discardRecording();
                router.replace('/');
              },
            },
          ]
        );
        return true;
      }

      if (stage === 'preview') {
        discardRecording();
        router.replace('/');
        return true;
      }

      router.replace('/');
      return true;
    };

    const sub = BackHandler.addEventListener('hardwareBackPress', onBackPress);
    return () => sub.remove();
  }, [stage, discardRecording]);

  const handleCancelFromActiveOrPaused = () => {
    Alert.alert(
      'Cancel Recording?',
      'Discard current recording session and return to start?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Discard',
          style: 'destructive',
          onPress: discardRecording,
        },
      ]
    );
  };

  return (
    <View style={styles.container}>
      {stage === 'idle' && (
        <RecordAudioScreen
          permissionStatus={permissionStatus}
          errorMessage={errorMessage}
          onRequestPermission={requestPermission}
          onStartRecording={startRecording}
          onBack={() => router.replace('/')}
        />
      )}

      {stage === 'recording' && (
        <ActiveRecordingScreen
          elapsedMs={elapsedMs}
          normalizedLevel={normalizedLevel}
          rawMetering={rawMetering}
          isMeteringAvailable={isMeteringAvailable}
          fileSizeBytes={fileSize}
          onPause={pauseRecording}
          onStop={stopRecording}
          onCancel={handleCancelFromActiveOrPaused}
        />
      )}

      {stage === 'paused' && (
        <PausedRecordingScreen
          elapsedMs={elapsedMs}
          fileSizeBytes={fileSize}
          onResume={resumeRecording}
          onStop={stopRecording}
          onCancel={handleCancelFromActiveOrPaused}
        />
      )}

      {stage === 'preview' && lastRecording && (
        <AudioPreviewScreen
          recording={lastRecording}
          onReRecord={discardRecording}
          onDelete={discardRecording}
          onDone={() => router.replace('/')}
        />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: BrandColors.primaryBackground,
  },
});

export default AudioRecordingCoordinator;

