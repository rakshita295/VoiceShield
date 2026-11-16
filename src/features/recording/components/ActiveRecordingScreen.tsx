import { LinearGradient } from 'expo-linear-gradient';
import React, { useEffect } from 'react';
import {
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import Animated, {
  Easing,
  FadeIn,
  FadeInDown,
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withSequence,
  withTiming,
} from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Svg, { Line, Path, Rect } from 'react-native-svg';

import { MeteringWaveform } from '@/features/recording/components/MeteringWaveform';
import { BrandColors, Spacing } from '@/constants/theme';

interface ActiveRecordingScreenProps {
  elapsedMs: number;
  normalizedLevel: number;
  rawMetering?: number;
  isMeteringAvailable: boolean;
  fileSizeBytes: number;
  onPause: () => void;
  onStop: () => void;
  onCancel: () => void;
}

export const ActiveRecordingScreen: React.FC<ActiveRecordingScreenProps> = ({
  elapsedMs,
  normalizedLevel,
  rawMetering,
  isMeteringAvailable,
  fileSizeBytes,
  onPause,
  onStop,
  onCancel,
}) => {
  const insets = useSafeAreaInsets();
  const pauseScale = useSharedValue(1);
  const stopScale = useSharedValue(1);

  // Live pulsing red recording indicator
  const livePulse = useSharedValue(1);
  useEffect(() => {
    livePulse.value = withRepeat(
      withSequence(
        withTiming(1.3, { duration: 600, easing: Easing.inOut(Easing.ease) }),
        withTiming(1.0, { duration: 600, easing: Easing.inOut(Easing.ease) })
      ),
      -1,
      true
    );
  }, [livePulse]);

  const animatedLiveStyle = useAnimatedStyle(() => ({
    transform: [{ scale: livePulse.value }],
    opacity: livePulse.value > 1.15 ? 1 : 0.75,
  }));

  const animatedPauseStyle = useAnimatedStyle(() => ({
    transform: [{ scale: pauseScale.value }],
  }));

  const animatedStopStyle = useAnimatedStyle(() => ({
    transform: [{ scale: stopScale.value }],
  }));

  // Format milliseconds into MM:SS.S
  const totalSeconds = Math.floor(elapsedMs / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  const tenths = Math.floor((elapsedMs % 1000) / 100);
  const formattedTime = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${tenths}`;

  // Format file size
  const formattedSize = fileSizeBytes > 0
    ? `${(fileSizeBytes / 1024).toFixed(1)} KB`
    : 'Buffering';

  return (
    <View style={styles.container}>
      {/* Top Header Bar */}
      <View style={[styles.topBar, { paddingTop: Math.max(insets.top, 24) }]}>
        <Pressable onPress={onCancel} hitSlop={12} style={styles.cancelButton}>
          <Svg width={22} height={22} viewBox="0 0 24 24" fill="none">
            <Line x1="18" y1="6" x2="6" y2="18" stroke="#FFFFFF" strokeWidth="2" strokeLinecap="round" />
            <Line x1="6" y1="6" x2="18" y2="18" stroke="#FFFFFF" strokeWidth="2" strokeLinecap="round" />
          </Svg>
        </Pressable>

        <Text style={styles.headerTitle}>Record Audio</Text>

        {/* Live Status Pill */}
        <View style={styles.liveBadge}>
          <Animated.View style={[styles.liveDot, animatedLiveStyle]} />
          <Text style={styles.liveText}>REC</Text>
        </View>
      </View>

      <View style={styles.content}>
        {/* Giant Glowing Duration Timer */}
        <Animated.View
          entering={FadeIn.duration(300)}
          style={styles.timerSection}>
          <Text style={styles.timerText}>{formattedTime}</Text>
          <Text style={styles.timerLabel}>RECORDING IN PROGRESS</Text>
        </Animated.View>

        {/* Real-time Decibel Audio Waveform */}
        <Animated.View
          entering={FadeInDown.delay(100).duration(400)}
          style={styles.waveformSection}>
          <MeteringWaveform
            normalizedLevel={normalizedLevel}
            rawMetering={rawMetering}
            isMeteringAvailable={isMeteringAvailable}
            isActive={true}
            isPaused={false}
            height={130}
          />
        </Animated.View>

        {/* Recording Progress Information Card */}
        <Animated.View
          entering={FadeInDown.delay(180).duration(400)}
          style={styles.infoCard}>
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Input Source:</Text>
            <Text style={styles.infoValue}>Device Microphone</Text>
          </View>
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Real-time Level:</Text>
            <Text style={[styles.infoValue, { color: BrandColors.cyanAccent }]}>
              {typeof rawMetering === 'number' ? `${rawMetering.toFixed(1)} dB` : 'Active'}
            </Text>
          </View>
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Buffer Stream:</Text>
            <Text style={styles.infoValue}>{formattedSize}</Text>
          </View>
        </Animated.View>

        {/* Control Action Buttons (Pause & Stop) */}
        <Animated.View
          entering={FadeInDown.delay(260).duration(400)}
          style={[styles.controlsRow, { paddingBottom: Math.max(insets.bottom, 24) + Spacing.two }]}>
          {/* Pause Button */}
          <Animated.View style={[styles.controlBtnWrapper, animatedPauseStyle]}>
            <Pressable
              onPress={onPause}
              onPressIn={() => (pauseScale.value = withTiming(0.92, { duration: 100 }))}
              onPressOut={() => (pauseScale.value = withTiming(1, { duration: 140 }))}
              style={styles.pauseBtn}>
              <View style={styles.pauseIconBadge}>
                <Svg width={22} height={22} viewBox="0 0 24 24" fill="none">
                  <Rect x="6" y="4" width="4" height="16" rx="1.5" fill="#F59E0B" />
                  <Rect x="14" y="4" width="4" height="16" rx="1.5" fill="#F59E0B" />
                </Svg>
              </View>
              <Text style={styles.pauseBtnText}>Pause</Text>
            </Pressable>
          </Animated.View>

          {/* Stop Button */}
          <Animated.View style={[styles.controlBtnWrapper, animatedStopStyle]}>
            <Pressable
              onPress={onStop}
              onPressIn={() => (stopScale.value = withTiming(0.92, { duration: 100 }))}
              onPressOut={() => (stopScale.value = withTiming(1, { duration: 140 }))}
              style={styles.stopBtn}>
              <LinearGradient
                colors={['#EF4444', '#DC2626', '#991B1B']}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={styles.stopGradient}>
                <View style={styles.stopIconBadge}>
                  <Svg width={18} height={18} viewBox="0 0 24 24" fill="none">
                    <Rect x="4" y="4" width="16" height="16" rx="3" fill="#FFFFFF" />
                  </Svg>
                </View>
                <Text style={styles.stopBtnText}>Stop & Preview</Text>
              </LinearGradient>
            </Pressable>
          </Animated.View>
        </Animated.View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: BrandColors.primaryBackground,
  },
  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: Spacing.four,
    paddingBottom: Spacing.two,
  },
  cancelButton: {
    padding: 6,
    borderRadius: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
  },
  headerTitle: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  liveBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    backgroundColor: 'rgba(239, 68, 68, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(239, 68, 68, 0.4)',
  },
  liveDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#EF4444',
  },
  liveText: {
    color: '#EF4444',
    fontSize: 10.5,
    fontWeight: '800',
    letterSpacing: 1.2,
  },
  content: {
    flex: 1,
    paddingHorizontal: Spacing.four,
    justifyContent: 'space-between',
  },
  timerSection: {
    alignItems: 'center',
    marginTop: Spacing.three,
  },
  timerText: {
    color: '#FFFFFF',
    fontSize: 54,
    fontWeight: '800',
    letterSpacing: 2,
    fontVariant: ['tabular-nums'],
  },
  timerLabel: {
    color: BrandColors.cyanAccent,
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1.5,
    marginTop: 4,
  },
  waveformSection: {
    alignItems: 'center',
    justifyContent: 'center',
    marginVertical: Spacing.two,
  },
  infoCard: {
    backgroundColor: BrandColors.cardBackground,
    borderRadius: 16,
    padding: Spacing.three,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    gap: 8,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  infoLabel: {
    color: BrandColors.secondaryText,
    fontSize: 12.5,
    fontWeight: '500',
  },
  infoValue: {
    color: '#FFFFFF',
    fontSize: 12.5,
    fontWeight: '600',
  },
  controlsRow: {
    flexDirection: 'row',
    gap: Spacing.three,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: Spacing.two,
  },
  controlBtnWrapper: {
    flex: 1,
  },
  pauseBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 16,
    borderRadius: 16,
    backgroundColor: 'rgba(245, 158, 11, 0.12)',
    borderWidth: 1.5,
    borderColor: 'rgba(245, 158, 11, 0.4)',
  },
  pauseIconBadge: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  pauseBtnText: {
    color: '#F59E0B',
    fontSize: 15.5,
    fontWeight: '700',
  },
  stopBtn: {
    borderRadius: 16,
    overflow: 'hidden',
  },
  stopGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 16,
  },
  stopIconBadge: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  stopBtnText: {
    color: '#FFFFFF',
    fontSize: 15.5,
    fontWeight: '800',
  },
});

