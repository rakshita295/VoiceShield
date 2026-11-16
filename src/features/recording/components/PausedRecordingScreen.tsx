import { LinearGradient } from 'expo-linear-gradient';
import React from 'react';
import {
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import Animated, {
  FadeIn,
  FadeInDown,
  useAnimatedStyle,
  useSharedValue,
  withTiming,
} from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Svg, { Path, Rect } from 'react-native-svg';

import { MeteringWaveform } from '@/features/recording/components/MeteringWaveform';
import { BrandColors, Spacing } from '@/constants/theme';

interface PausedRecordingScreenProps {
  elapsedMs: number;
  fileSizeBytes: number;
  onResume: () => void;
  onStop: () => void;
  onCancel: () => void;
}

export const PausedRecordingScreen: React.FC<PausedRecordingScreenProps> = ({
  elapsedMs,
  fileSizeBytes,
  onResume,
  onStop,
  onCancel,
}) => {
  const insets = useSafeAreaInsets();
  const resumeScale = useSharedValue(1);
  const stopScale = useSharedValue(1);

  const animatedResumeStyle = useAnimatedStyle(() => ({
    transform: [{ scale: resumeScale.value }],
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

  const formattedSize = fileSizeBytes > 0
    ? `${(fileSizeBytes / 1024).toFixed(1)} KB`
    : 'Preserved';

  return (
    <View style={styles.container}>
      {/* Top Header Bar */}
      <View style={[styles.topBar, { paddingTop: Math.max(insets.top, 24) }]}>
        <Pressable onPress={onCancel} hitSlop={12} style={styles.cancelButton}>
          <Text style={styles.cancelText}>Cancel</Text>
        </Pressable>

        <Text style={styles.headerTitle}>Record Audio</Text>

        {/* Paused Status Badge */}
        <View style={styles.pausedBadge}>
          <View style={styles.pausedDot} />
          <Text style={styles.pausedText}>PAUSED</Text>
        </View>
      </View>

      <View style={styles.content}>
        {/* Frozen Timer */}
        <Animated.View
          entering={FadeIn.duration(300)}
          style={styles.timerSection}>
          <Text style={styles.timerText}>{formattedTime}</Text>
          <Text style={styles.timerLabel}>SESSION TEMPORARILY PAUSED</Text>
        </Animated.View>

        {/* Subdued Static Waveform */}
        <Animated.View
          entering={FadeInDown.delay(100).duration(400)}
          style={styles.waveformSection}>
          <MeteringWaveform
            normalizedLevel={0}
            isMeteringAvailable={false}
            isActive={false}
            isPaused={true}
            height={130}
          />
        </Animated.View>

        {/* Informational Guidance Card */}
        <Animated.View
          entering={FadeInDown.delay(180).duration(400)}
          style={styles.infoCard}>
          <View style={styles.infoIconBox}>
            <Svg width={24} height={24} viewBox="0 0 24 24" fill="none">
              <Rect x="6" y="4" width="4" height="16" rx="1.5" fill={BrandColors.cyanAccent} />
              <Rect x="14" y="4" width="4" height="16" rx="1.5" fill={BrandColors.cyanAccent} />
            </Svg>
          </View>
          <View style={styles.infoTextGroup}>
            <Text style={styles.infoTitle}>Recording Paused</Text>
            <Text style={styles.infoDesc}>
              Audio session is preserved ({formattedSize}). Tap Resume to continue speaking or Stop to finalize.
            </Text>
          </View>
        </Animated.View>

        {/* Action Controls (Resume & Stop) */}
        <Animated.View
          entering={FadeInDown.delay(260).duration(400)}
          style={[styles.controlsRow, { paddingBottom: Math.max(insets.bottom, 24) + Spacing.two }]}>
          {/* Resume Button */}
          <Animated.View style={[styles.controlBtnWrapper, animatedResumeStyle]}>
            <Pressable
              onPress={onResume}
              onPressIn={() => (resumeScale.value = withTiming(0.92, { duration: 100 }))}
              onPressOut={() => (resumeScale.value = withTiming(1, { duration: 140 }))}
              style={styles.resumeBtn}>
              <LinearGradient
                colors={[BrandColors.electricBlue, BrandColors.cyanAccent]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
                style={styles.resumeGradient}>
                <Svg width={20} height={20} viewBox="0 0 24 24" fill="none">
                  <Path d="M5 3L19 12L5 21V3Z" fill="#FFFFFF" />
                </Svg>
                <Text style={styles.resumeBtnText}>Resume</Text>
              </LinearGradient>
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
                <Svg width={18} height={18} viewBox="0 0 24 24" fill="none">
                  <Rect x="4" y="4" width="16" height="16" rx="3" fill="#FFFFFF" />
                </Svg>
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
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
  },
  cancelText: {
    color: BrandColors.secondaryText,
    fontSize: 12.5,
    fontWeight: '600',
  },
  headerTitle: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  pausedBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    backgroundColor: 'rgba(245, 158, 11, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(245, 158, 11, 0.4)',
  },
  pausedDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#F59E0B',
  },
  pausedText: {
    color: '#F59E0B',
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
    color: '#D1D5DB',
    fontSize: 54,
    fontWeight: '800',
    letterSpacing: 2,
    fontVariant: ['tabular-nums'],
  },
  timerLabel: {
    color: '#F59E0B',
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
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    backgroundColor: BrandColors.cardBackground,
    borderRadius: 16,
    padding: Spacing.three,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
  },
  infoIconBox: {
    width: 44,
    height: 44,
    borderRadius: 12,
    backgroundColor: 'rgba(0, 217, 255, 0.1)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  infoTextGroup: {
    flex: 1,
  },
  infoTitle: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '700',
  },
  infoDesc: {
    color: BrandColors.secondaryText,
    fontSize: 12,
    fontWeight: '500',
    marginTop: 2,
    lineHeight: 18,
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
  resumeBtn: {
    borderRadius: 16,
    overflow: 'hidden',
  },
  resumeGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 16,
  },
  resumeBtnText: {
    color: '#070817',
    fontSize: 15.5,
    fontWeight: '800',
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
  stopBtnText: {
    color: '#FFFFFF',
    fontSize: 15.5,
    fontWeight: '800',
  },
});

