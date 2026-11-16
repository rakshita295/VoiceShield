import { LinearGradient } from 'expo-linear-gradient';
import React, { useState } from 'react';
import {
  Alert,
  Dimensions,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import Animated, {
  FadeInDown,
  useAnimatedStyle,
  useSharedValue,
  withTiming,
} from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Svg, {
  Defs,
  LinearGradient as SvgLinearGradient,
  Path,
  Rect,
  Stop,
} from 'react-native-svg';

import { useVoicePlayer } from '@/features/recording/hooks/useVoicePlayer';
import { UploadedAudioMeta } from '@/features/upload/types/uploadTypes';
import { BrandColors, Spacing } from '@/constants/theme';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

interface UploadedAudioPreviewScreenProps {
  audioMeta: UploadedAudioMeta;
  onReplace: () => void;
  onDiscard: () => void;
  onDone: () => void;
}

export const UploadedAudioPreviewScreen: React.FC<UploadedAudioPreviewScreenProps> = ({
  audioMeta,
  onReplace,
  onDiscard,
  onDone,
}) => {
  const insets = useSafeAreaInsets();
  const [showAiModal, setShowAiModal] = useState(false);

  // Audio player hook using expo-audio
  const { isPlaying, currentTime, duration, togglePlayPause, seekTo } =
    useVoicePlayer(audioMeta.uri);

  const analyzeBtnScale = useSharedValue(1);
  const replaceBtnScale = useSharedValue(1);

  const animatedAnalyzeStyle = useAnimatedStyle(() => ({
    transform: [{ scale: analyzeBtnScale.value }],
  }));

  const animatedReplaceStyle = useAnimatedStyle(() => ({
    transform: [{ scale: replaceBtnScale.value }],
  }));

  const totalDuration = duration > 0 ? duration : Math.max(1, audioMeta.durationSec);
  const currentProgress = Math.min(1, Math.max(0, currentTime / totalDuration));

  const formatSec = (sec: number) => {
    const mins = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${String(mins).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  };

  const handleProgressBarPress = (e: any) => {
    const { locationX } = e.nativeEvent;
    const barWidth = SCREEN_WIDTH - Spacing.four * 2 - 32;
    if (barWidth > 0) {
      const ratio = Math.min(1, Math.max(0, locationX / barWidth));
      seekTo(ratio * totalDuration);
    }
  };

  const handleDiscardPress = () => {
    Alert.alert(
      'Discard Audio File?',
      'The selected audio will be removed from memory and you will return to the dashboard.',
      [
        { text: 'Keep', style: 'cancel' },
        {
          text: 'Discard',
          style: 'destructive',
          onPress: onDiscard,
        },
      ]
    );
  };

  return (
    <View style={styles.container}>
      {/* Top Header Bar */}
      <View style={[styles.topBar, { paddingTop: Math.max(insets.top, 24) }]}>
        <Pressable onPress={onDone} hitSlop={12} style={styles.backButton}>
          <Svg width={22} height={22} viewBox="0 0 24 24" fill="none">
            <Path
              d="M19 12H5M5 12L12 19M5 12L12 5"
              stroke="#FFFFFF"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </Svg>
        </Pressable>

        <Text style={styles.headerTitle}>Uploaded Audio</Text>

        <View style={styles.successBadge}>
          <Text style={styles.successBadgeText}>VALIDATED</Text>
        </View>
      </View>

      <ScrollView
        contentContainerStyle={[
          styles.scrollContent,
          { paddingBottom: Math.max(insets.bottom, 24) + Spacing.four },
        ]}
        showsVerticalScrollIndicator={false}>
        {/* Subtitle */}
        <Animated.View
          entering={FadeInDown.duration(400)}
          style={styles.headingSection}>
          <Text style={styles.previewHeading}>File Ready for Analysis</Text>
          <Text style={styles.previewSubtext}>
            The selected audio has been verified and loaded into local sandbox memory.
          </Text>
        </Animated.View>

        {/* Audio Player Card */}
        <Animated.View
          entering={FadeInDown.delay(120).duration(450)}
          style={styles.playerCard}>
          {/* Waveform Art Header */}
          <View style={styles.waveformPreviewBox}>
            <Svg width={260} height={50} viewBox="0 0 260 50">
              <Defs>
                <SvgLinearGradient id="uploadWaveGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <Stop offset="0%" stopColor={BrandColors.electricBlue} />
                  <Stop offset="100%" stopColor={BrandColors.cyanAccent} />
                </SvgLinearGradient>
              </Defs>
              {Array.from({ length: 26 }).map((_, i) => {
                const h = 10 + Math.sin(i * 0.8) * 16 + (i % 3) * 6;
                const isPassed = (i / 26) <= currentProgress;
                return (
                  <Rect
                    key={i}
                    x={i * 10}
                    y={(50 - h) / 2}
                    width={5}
                    height={h}
                    rx={2.5}
                    fill="url(#uploadWaveGrad)"
                    opacity={isPassed ? 1 : 0.3}
                  />
                );
              })}
            </Svg>
          </View>

          {/* Interactive Scrubbable Progress Bar */}
          <Pressable onPress={handleProgressBarPress} style={styles.progressBarWrapper}>
            <View style={styles.progressBarTrack}>
              <View
                style={[
                  styles.progressBarFill,
                  { width: `${Math.round(currentProgress * 100)}%` },
                ]}
              />
            </View>
          </Pressable>

          {/* Time Labels */}
          <View style={styles.timeRow}>
            <Text style={styles.timeCurrent}>{formatSec(currentTime)}</Text>
            <Text style={styles.timeDuration}>{formatSec(totalDuration)}</Text>
          </View>

          {/* Big Play / Pause Button */}
          <View style={styles.playButtonRow}>
            <Pressable onPress={togglePlayPause} style={styles.playBtnCircle}>
              <LinearGradient
                colors={[BrandColors.electricBlue, BrandColors.cyanAccent]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={styles.playBtnGradient}>
                {isPlaying ? (
                  <Svg width={24} height={24} viewBox="0 0 24 24" fill="none">
                    <Rect x="6" y="4" width="4" height="16" rx="1.5" fill="#070817" />
                    <Rect x="14" y="4" width="4" height="16" rx="1.5" fill="#070817" />
                  </Svg>
                ) : (
                  <Svg width={24} height={24} viewBox="0 0 24 24" fill="none">
                    <Path d="M7 4L19 12L7 20V4Z" fill="#070817" />
                  </Svg>
                )}
              </LinearGradient>
            </Pressable>
          </View>
        </Animated.View>

        {/* Audio File Metadata Card */}
        <Animated.View
          entering={FadeInDown.delay(200).duration(450)}
          style={styles.metaCard}>
          <Text style={styles.metaHeading}>File Details</Text>

          <View style={styles.metaRow}>
            <Text style={styles.metaLabel}>File Name:</Text>
            <Text style={styles.metaValue} numberOfLines={1}>
              {audioMeta.fileName}
            </Text>
          </View>
          <View style={styles.metaRow}>
            <Text style={styles.metaLabel}>Duration:</Text>
            <Text style={styles.metaValue}>{formatSec(totalDuration)}</Text>
          </View>
          <View style={styles.metaRow}>
            <Text style={styles.metaLabel}>File Size:</Text>
            <Text style={styles.metaValue}>
              {(audioMeta.fileSizeBytes / 1024).toFixed(1)} KB
            </Text>
          </View>
          <View style={styles.metaRow}>
            <Text style={styles.metaLabel}>Format:</Text>
            <Text style={styles.metaValue}>
              {audioMeta.extension.toUpperCase().replace('.', '')} Audio
            </Text>
          </View>
          <View style={styles.metaRow}>
            <Text style={styles.metaLabel}>Storage:</Text>
            <Text style={[styles.metaValue, { color: BrandColors.cyanAccent }]}>
              Local Cache Sandbox
            </Text>
          </View>
        </Animated.View>

        {/* On-Device Privacy Guarantee */}
        <View style={styles.privacyNoteBox}>
          <Svg width={18} height={18} viewBox="0 0 24 24" fill="none">
            <Path
              d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"
              stroke={BrandColors.cyanAccent}
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </Svg>
          <Text style={styles.privacyNoteText}>
            100% On-Device • No Cloud Uploads • Private Audio Verification
          </Text>
        </View>

        {/* Primary Action: Verify Voice Authenticity */}
        <Animated.View
          entering={FadeInDown.delay(280).duration(450)}
          style={[styles.btnWrapper, animatedAnalyzeStyle]}>
          <Pressable
            onPress={() => setShowAiModal(true)}
            onPressIn={() => (analyzeBtnScale.value = withTiming(0.96, { duration: 100 }))}
            onPressOut={() => (analyzeBtnScale.value = withTiming(1, { duration: 150 }))}
            style={styles.analyzeBtn}>
            <LinearGradient
              colors={[BrandColors.primaryPurple, BrandColors.electricBlue, BrandColors.cyanAccent]}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={styles.analyzeGradient}>
              <Svg width={22} height={22} viewBox="0 0 24 24" fill="none">
                <Path
                  d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"
                  fill="#FFFFFF"
                />
              </Svg>
              <Text style={styles.analyzeBtnText}>Verify Voice Authenticity</Text>
            </LinearGradient>
          </Pressable>
        </Animated.View>

        {/* Secondary Actions: Replace File & Discard */}
        <Animated.View
          entering={FadeInDown.delay(360).duration(450)}
          style={styles.secondaryActionsRow}>
          {/* Replace Button */}
          <Animated.View style={[styles.secondaryBtnWrapper, animatedReplaceStyle]}>
            <Pressable
              onPress={onReplace}
              onPressIn={() => (replaceBtnScale.value = withTiming(0.94, { duration: 100 }))}
              onPressOut={() => (replaceBtnScale.value = withTiming(1, { duration: 140 }))}
              style={styles.replaceBtn}>
              <Svg width={18} height={18} viewBox="0 0 24 24" fill="none">
                <Path
                  d="M1 4V10H7M23 20V14H17M20.49 9A9 9 0 0 0 5.64 5.64L1 10M23 14L18.36 18.36A9 9 0 0 1 3.51 15"
                  stroke={BrandColors.electricBlue}
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </Svg>
              <Text style={styles.replaceBtnText}>Choose Another</Text>
            </Pressable>
          </Animated.View>

          {/* Discard Button */}
          <Pressable onPress={handleDiscardPress} style={styles.discardBtn}>
            <Svg width={18} height={18} viewBox="0 0 24 24" fill="none">
              <Path
                d="M3 6H5H21M19 6V20C19 21.1046 18.1046 22 17 22H7C5.89543 22 5 21.1046 5 20V6M8 6V4C8 2.89543 8.89543 2 10 2H14C15.1046 2 16 2.89543 16 4V6"
                stroke="#EF4444"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </Svg>
            <Text style={styles.discardBtnText}>Discard</Text>
          </Pressable>
        </Animated.View>
      </ScrollView>

      {/* Phase 5 AI Modal Preview */}
      {showAiModal && (
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalIconBox}>
              <Svg width={32} height={32} viewBox="0 0 24 24" fill="none">
                <Path
                  d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"
                  stroke={BrandColors.cyanAccent}
                  strokeWidth="2"
                />
                <Path d="M9 12L11 14L15 10" stroke={BrandColors.cyanAccent} strokeWidth="2" />
              </Svg>
            </View>
            <Text style={styles.modalTitle}>AI Verification Pipeline</Text>
            <Text style={styles.modalBody}>
              The uploaded audio file is verified and ready. In Phase 5, this audio will be preprocessed to match our custom neural model requirements and evaluated locally on your device for AI voice manipulation.
            </Text>
            <Pressable onPress={() => setShowAiModal(false)} style={styles.modalCloseBtn}>
              <Text style={styles.modalCloseText}>Got It</Text>
            </Pressable>
          </View>
        </View>
      )}
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
  backButton: {
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
  successBadge: {
    backgroundColor: 'rgba(0, 217, 255, 0.15)',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(0, 217, 255, 0.3)',
  },
  successBadgeText: {
    color: BrandColors.cyanAccent,
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 1,
  },
  scrollContent: {
    paddingHorizontal: Spacing.four,
  },
  headingSection: {
    marginTop: Spacing.two,
    alignItems: 'center',
  },
  previewHeading: {
    color: '#FFFFFF',
    fontSize: 24,
    fontWeight: '800',
    textAlign: 'center',
  },
  previewSubtext: {
    color: BrandColors.secondaryText,
    fontSize: 13,
    fontWeight: '500',
    textAlign: 'center',
    marginTop: Spacing.one,
    lineHeight: 18,
    paddingHorizontal: Spacing.two,
  },
  playerCard: {
    backgroundColor: BrandColors.cardBackground,
    borderRadius: 20,
    padding: Spacing.three,
    marginTop: Spacing.three,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    alignItems: 'center',
  },
  waveformPreviewBox: {
    height: 50,
    alignItems: 'center',
    justifyContent: 'center',
    marginVertical: Spacing.two,
  },
  progressBarWrapper: {
    width: '100%',
    paddingVertical: 8,
  },
  progressBarTrack: {
    height: 6,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 3,
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    backgroundColor: BrandColors.cyanAccent,
    borderRadius: 3,
  },
  timeRow: {
    width: '100%',
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 4,
  },
  timeCurrent: {
    color: BrandColors.cyanAccent,
    fontSize: 12,
    fontWeight: '700',
  },
  timeDuration: {
    color: BrandColors.secondaryText,
    fontSize: 12,
    fontWeight: '600',
  },
  playButtonRow: {
    marginTop: Spacing.two,
    alignItems: 'center',
  },
  playBtnCircle: {
    borderRadius: 32,
    overflow: 'hidden',
  },
  playBtnGradient: {
    width: 64,
    height: 64,
    borderRadius: 32,
    alignItems: 'center',
    justifyContent: 'center',
  },
  metaCard: {
    backgroundColor: BrandColors.cardBackground,
    borderRadius: 16,
    padding: Spacing.three,
    marginTop: Spacing.three,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    gap: 8,
  },
  metaHeading: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '700',
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.06)',
    paddingBottom: Spacing.two,
    marginBottom: 4,
  },
  metaRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  metaLabel: {
    color: BrandColors.secondaryText,
    fontSize: 12,
    fontWeight: '500',
  },
  metaValue: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '600',
    maxWidth: '65%',
  },
  privacyNoteBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginVertical: Spacing.three,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 12,
    backgroundColor: 'rgba(0, 217, 255, 0.06)',
    borderWidth: 1,
    borderColor: 'rgba(0, 217, 255, 0.15)',
  },
  privacyNoteText: {
    color: BrandColors.secondaryText,
    fontSize: 11,
    fontWeight: '600',
    flex: 1,
  },
  btnWrapper: {
    width: '100%',
  },
  analyzeBtn: {
    borderRadius: 16,
    overflow: 'hidden',
  },
  analyzeGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    paddingVertical: 16,
  },
  analyzeBtnText: {
    color: '#FFFFFF',
    fontSize: 16.5,
    fontWeight: '800',
    letterSpacing: 0.6,
  },
  secondaryActionsRow: {
    flexDirection: 'row',
    gap: Spacing.three,
    marginTop: Spacing.three,
  },
  secondaryBtnWrapper: {
    flex: 1,
  },
  replaceBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 14,
    borderRadius: 14,
    backgroundColor: 'rgba(22, 139, 255, 0.1)',
    borderWidth: 1.5,
    borderColor: 'rgba(22, 139, 255, 0.35)',
  },
  replaceBtnText: {
    color: BrandColors.electricBlue,
    fontSize: 14.5,
    fontWeight: '700',
  },
  discardBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 14,
    paddingHorizontal: 20,
    borderRadius: 14,
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    borderWidth: 1.5,
    borderColor: 'rgba(239, 68, 68, 0.35)',
  },
  discardBtnText: {
    color: '#EF4444',
    fontSize: 14.5,
    fontWeight: '700',
  },
  modalOverlay: {
    ...StyleSheet.absoluteFill,
    backgroundColor: 'rgba(0, 0, 0, 0.75)',
    alignItems: 'center',
    justifyContent: 'center',
    padding: Spacing.four,
    zIndex: 999,
  },
  modalContent: {
    width: '100%',
    backgroundColor: BrandColors.cardBackground,
    borderRadius: 20,
    padding: Spacing.four,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(0, 217, 255, 0.3)',
  },
  modalIconBox: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: 'rgba(0, 217, 255, 0.12)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.two,
  },
  modalTitle: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '800',
    marginBottom: Spacing.two,
    textAlign: 'center',
  },
  modalBody: {
    color: BrandColors.secondaryText,
    fontSize: 13,
    fontWeight: '500',
    lineHeight: 20,
    textAlign: 'center',
    marginBottom: Spacing.three,
  },
  modalCloseBtn: {
    width: '100%',
    paddingVertical: 12,
    borderRadius: 12,
    backgroundColor: BrandColors.electricBlue,
    alignItems: 'center',
  },
  modalCloseText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '700',
  },
});

