import { router } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import React, { useEffect } from 'react';
import {
  ActivityIndicator,
  Alert,
  BackHandler,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import Animated, { FadeInDown } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Svg, { Circle, Line, Path, Rect } from 'react-native-svg';

import { VoiceShieldLogo } from '@/components/branding/VoiceShieldLogo';
import { UploadedAudioPreviewScreen } from '@/features/upload/components/UploadedAudioPreviewScreen';
import { useAudioFilePicker } from '@/features/upload/hooks/useAudioFilePicker';
import { SUPPORTED_AUDIO_FORMATS } from '@/features/upload/types/uploadTypes';
import { BrandColors, Spacing } from '@/constants/theme';

export const AudioUploadCoordinator: React.FC = () => {
  const insets = useSafeAreaInsets();
  const {
    selectedAudio,
    isValidating,
    validationError,
    pickAudioFile,
    removeAudioFile,
    clearError,
  } = useAudioFilePicker();

  // Android hardware back button handler
  useEffect(() => {
    const onBackPress = () => {
      if (selectedAudio) {
        Alert.alert(
          'Discard Audio?',
          'Discard this selected audio file and return to the dashboard?',
          [
            { text: 'Keep', style: 'cancel' },
            {
              text: 'Discard',
              style: 'destructive',
              onPress: () => {
                removeAudioFile();
                router.replace('/');
              },
            },
          ]
        );
        return true;
      }

      router.replace('/');
      return true;
    };

    const sub = BackHandler.addEventListener('hardwareBackPress', onBackPress);
    return () => sub.remove();
  }, [selectedAudio, removeAudioFile]);

  const handleDiscard = () => {
    removeAudioFile();
    router.replace('/');
  };

  const handleDone = () => {
    removeAudioFile();
    router.replace('/');
  };

  // If a file is selected and validated, show the preview screen
  if (selectedAudio) {
    return (
      <UploadedAudioPreviewScreen
        audioMeta={selectedAudio}
        onReplace={pickAudioFile}
        onDiscard={handleDiscard}
        onDone={handleDone}
      />
    );
  }

  return (
    <View style={styles.container}>
      {/* Top Header Bar */}
      <View style={[styles.topBar, { paddingTop: Math.max(insets.top, 24) }]}>
        <Pressable onPress={() => router.replace('/')} hitSlop={12} style={styles.backButton}>
          <Svg width={24} height={24} viewBox="0 0 24 24" fill="none">
            <Path
              d="M19 12H5M5 12L12 19M5 12L12 5"
              stroke="#FFFFFF"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </Svg>
        </Pressable>

        <View style={styles.headerBrandRow}>
          <VoiceShieldLogo size={30} animated={false} />
          <Text style={styles.headerBrandText}>VoiceShield</Text>
        </View>

        <View style={styles.headerRightPlaceholder} />
      </View>

      <ScrollView
        contentContainerStyle={[
          styles.scrollContent,
          { paddingBottom: Math.max(insets.bottom, 24) + Spacing.four },
        ]}
        showsVerticalScrollIndicator={false}>
        {/* Heading */}
        <Animated.View
          entering={FadeInDown.duration(400)}
          style={styles.headingSection}>
          <Text style={styles.mainHeading}>Upload Audio File</Text>
          <Text style={styles.descriptionText}>
            Select a pre-recorded audio sample from your device for deepfake and voice manipulation detection.
          </Text>
        </Animated.View>

        {/* Validation Error Alert if triggered */}
        {validationError && (
          <Animated.View
            entering={FadeInDown.duration(300)}
            style={styles.errorCard}>
            <View style={styles.errorHeaderRow}>
              <View style={styles.errorIconCircle}>
                <Svg width={18} height={18} viewBox="0 0 24 24" fill="none">
                  <Circle cx="12" cy="12" r="10" stroke="#EF4444" strokeWidth="2" />
                  <Line x1="12" y1="8" x2="12" y2="12" stroke="#EF4444" strokeWidth="2" strokeLinecap="round" />
                  <Circle cx="12" cy="16" r="1" fill="#EF4444" />
                </Svg>
              </View>
              <Text style={styles.errorTitle}>{validationError.title}</Text>
            </View>
            <Text style={styles.errorMessage}>{validationError.message}</Text>
            <Pressable onPress={clearError} style={styles.errorDismissBtn}>
              <Text style={styles.errorDismissText}>Dismiss</Text>
            </Pressable>
          </Animated.View>
        )}

        {/* Graphic Upload Dropzone Card */}
        <Animated.View
          entering={FadeInDown.delay(120).duration(450)}
          style={styles.uploadCard}>
          <View style={styles.uploadIconBadge}>
            <Svg width={36} height={36} viewBox="0 0 24 24" fill="none">
              <Path
                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
                stroke={BrandColors.cyanAccent}
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </Svg>
          </View>

          <Text style={styles.uploadCardTitle}>Select Voice Recording</Text>
          <Text style={styles.uploadCardDesc}>
            Tap below to open the Android system file manager and choose an audio file.
          </Text>

          {isValidating ? (
            <View style={styles.validatingBox}>
              <ActivityIndicator size="small" color={BrandColors.cyanAccent} />
              <Text style={styles.validatingText}>Verifying container & native codecs...</Text>
            </View>
          ) : (
            <Pressable
              onPress={pickAudioFile}
              style={styles.pickButton}>
              <LinearGradient
                colors={[BrandColors.primaryPurple, BrandColors.electricBlue, BrandColors.cyanAccent]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
                style={styles.pickBtnGradient}>
                <Svg width={20} height={20} viewBox="0 0 24 24" fill="none">
                  <Path
                    d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"
                    stroke="#FFFFFF"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <Path d="M14 2v6h6" stroke="#FFFFFF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </Svg>
                <Text style={styles.pickBtnText}>Browse Audio Files</Text>
              </LinearGradient>
            </Pressable>
          )}
        </Animated.View>

        {/* Supported Audio Formats & Limits Section */}
        <Animated.View
          entering={FadeInDown.delay(200).duration(450)}
          style={styles.infoSection}>
          <Text style={styles.infoSectionHeading}>Supported Formats & Constraints</Text>

          <View style={styles.formatGrid}>
            {SUPPORTED_AUDIO_FORMATS.map((fmt) => (
              <View key={fmt.extension} style={styles.formatChip}>
                <Text style={styles.formatChipLabel}>{fmt.label}</Text>
                <Text style={styles.formatChipDesc} numberOfLines={1}>
                  {fmt.description}
                </Text>
              </View>
            ))}
          </View>

          <View style={styles.limitsCard}>
            <View style={styles.limitRow}>
              <Text style={styles.limitLabel}>Maximum File Size:</Text>
              <Text style={styles.limitVal}>25 MB</Text>
            </View>
            <View style={styles.limitRow}>
              <Text style={styles.limitLabel}>Recommended Duration:</Text>
              <Text style={styles.limitVal}>3.0s – 60s (Max 5.0m)</Text>
            </View>
            <View style={styles.limitRow}>
              <Text style={styles.limitLabel}>Processing Mode:</Text>
              <Text style={[styles.limitVal, { color: BrandColors.cyanAccent }]}>
                100% Local On-Device
              </Text>
            </View>
          </View>
        </Animated.View>
      </ScrollView>
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
    backgroundColor: 'rgba(255, 255, 255, 0.06)',
  },
  headerBrandRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  headerBrandText: {
    color: '#FFFFFF',
    fontSize: 17,
    fontWeight: '800',
    letterSpacing: 0.6,
  },
  headerRightPlaceholder: {
    width: 36,
  },
  scrollContent: {
    paddingHorizontal: Spacing.four,
  },
  headingSection: {
    marginTop: Spacing.three,
    alignItems: 'center',
  },
  mainHeading: {
    color: '#FFFFFF',
    fontSize: 26,
    fontWeight: '800',
    letterSpacing: 0.5,
    textAlign: 'center',
  },
  descriptionText: {
    color: BrandColors.secondaryText,
    fontSize: 13.5,
    fontWeight: '500',
    textAlign: 'center',
    marginTop: Spacing.one,
    lineHeight: 20,
    paddingHorizontal: Spacing.two,
  },
  errorCard: {
    backgroundColor: 'rgba(239, 68, 68, 0.12)',
    borderRadius: 14,
    borderWidth: 1,
    borderColor: 'rgba(239, 68, 68, 0.35)',
    padding: Spacing.three,
    marginTop: Spacing.three,
  },
  errorHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginBottom: 6,
  },
  errorIconCircle: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: 'rgba(239, 68, 68, 0.2)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  errorTitle: {
    color: '#EF4444',
    fontSize: 14,
    fontWeight: '700',
  },
  errorMessage: {
    color: BrandColors.secondaryText,
    fontSize: 12.5,
    fontWeight: '500',
    lineHeight: 18,
  },
  errorDismissBtn: {
    marginTop: 8,
    alignSelf: 'flex-end',
    paddingVertical: 4,
    paddingHorizontal: 8,
  },
  errorDismissText: {
    color: '#EF4444',
    fontSize: 12,
    fontWeight: '700',
  },
  uploadCard: {
    backgroundColor: BrandColors.cardBackground,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: 'rgba(0, 217, 255, 0.2)',
    padding: Spacing.four,
    alignItems: 'center',
    marginTop: Spacing.four,
  },
  uploadIconBadge: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: 'rgba(0, 217, 255, 0.1)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.three,
    borderWidth: 1.5,
    borderColor: 'rgba(0, 217, 255, 0.25)',
  },
  uploadCardTitle: {
    color: '#FFFFFF',
    fontSize: 17,
    fontWeight: '800',
    marginBottom: 4,
  },
  uploadCardDesc: {
    color: BrandColors.secondaryText,
    fontSize: 12.5,
    fontWeight: '500',
    textAlign: 'center',
    lineHeight: 18,
    marginBottom: Spacing.four,
    paddingHorizontal: Spacing.two,
  },
  validatingBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 14,
  },
  validatingText: {
    color: BrandColors.cyanAccent,
    fontSize: 13,
    fontWeight: '600',
  },
  pickButton: {
    width: '100%',
    borderRadius: 16,
    overflow: 'hidden',
  },
  pickBtnGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    gap: 10,
  },
  pickBtnText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  infoSection: {
    marginTop: Spacing.four,
  },
  infoSectionHeading: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '700',
    marginBottom: Spacing.two,
  },
  formatGrid: {
    gap: 8,
    marginBottom: Spacing.three,
  },
  formatChip: {
    backgroundColor: 'rgba(20, 26, 54, 0.7)',
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 9,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.06)',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  formatChipLabel: {
    color: BrandColors.cyanAccent,
    fontSize: 12,
    fontWeight: '800',
    width: 44,
  },
  formatChipDesc: {
    color: BrandColors.secondaryText,
    fontSize: 11.5,
    fontWeight: '500',
    flex: 1,
  },
  limitsCard: {
    backgroundColor: BrandColors.cardBackground,
    borderRadius: 14,
    padding: Spacing.three,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    gap: 6,
  },
  limitRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  limitLabel: {
    color: BrandColors.secondaryText,
    fontSize: 12,
    fontWeight: '500',
  },
  limitVal: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '600',
  },
});

export default AudioUploadCoordinator;

