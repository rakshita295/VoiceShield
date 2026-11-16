import { LinearGradient } from 'expo-linear-gradient';
import React from 'react';
import {
  Dimensions,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import Animated, {
  Easing,
  FadeInDown,
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withSequence,
  withTiming,
} from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Svg, {
  Circle,
  Defs,
  Line,
  LinearGradient as SvgLinearGradient,
  Path,
  Rect,
  Stop,
} from 'react-native-svg';

import { VoiceShieldLogo } from '@/components/branding/VoiceShieldLogo';
import { DEFAULT_AUDIO_CONFIG } from '@/features/recording/data/recordingConfig';
import { MicrophonePermissionStatus } from '@/features/recording/types/recordingTypes';
import { BrandColors, Spacing } from '@/constants/theme';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

interface RecordAudioScreenProps {
  permissionStatus: MicrophonePermissionStatus;
  errorMessage: string | null;
  onRequestPermission: () => Promise<boolean>;
  onStartRecording: () => void;
  onBack: () => void;
}

export const RecordAudioScreen: React.FC<RecordAudioScreenProps> = ({
  permissionStatus,
  errorMessage,
  onRequestPermission,
  onStartRecording,
  onBack,
}) => {
  const insets = useSafeAreaInsets();
  const startBtnScale = useSharedValue(1);

  // Subtle breathing animation for microphone illustration
  const micPulse = useSharedValue(1);
  React.useEffect(() => {
    micPulse.value = withRepeat(
      withSequence(
        withTiming(1.06, { duration: 2000, easing: Easing.inOut(Easing.ease) }),
        withTiming(1.0, { duration: 2000, easing: Easing.inOut(Easing.ease) })
      ),
      -1,
      true
    );
  }, [micPulse]);

  const animatedMicStyle = useAnimatedStyle(() => ({
    transform: [{ scale: micPulse.value }],
  }));

  const animatedBtnStyle = useAnimatedStyle(() => ({
    transform: [{ scale: startBtnScale.value }],
  }));

  const renderPermissionBadge = () => {
    switch (permissionStatus) {
      case 'granted':
        return (
          <View style={[styles.permCard, styles.permGranted]}>
            <View style={styles.permIconBadge}>
              <Svg width={18} height={18} viewBox="0 0 24 24" fill="none">
                <Path
                  d="M5 13L9 17L19 7"
                  stroke={BrandColors.cyanAccent}
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </Svg>
            </View>
            <View style={styles.permTextGroup}>
              <Text style={styles.permTitle}>Microphone Permission Granted</Text>
              <Text style={styles.permSubtitle}>Ready for high-fidelity voice capture</Text>
            </View>
          </View>
        );
      case 'denied':
        return (
          <Pressable onPress={onRequestPermission} style={[styles.permCard, styles.permDenied]}>
            <View style={[styles.permIconBadge, { backgroundColor: 'rgba(239, 68, 68, 0.2)' }]}>
              <Svg width={18} height={18} viewBox="0 0 24 24" fill="none">
                <Path
                  d="M12 8V12M12 16H12.01M22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2C17.5228 2 22 6.47715 22 12Z"
                  stroke="#EF4444"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </Svg>
            </View>
            <View style={styles.permTextGroup}>
              <Text style={[styles.permTitle, { color: '#EF4444' }]}>Microphone Access Denied</Text>
              <Text style={styles.permSubtitle}>Tap here to grant access in system settings</Text>
            </View>
          </Pressable>
        );
      case 'not_granted':
      case 'checking':
      default:
        return (
          <View style={[styles.permCard, styles.permPending]}>
            <View style={styles.permIconBadge}>
              <Svg width={18} height={18} viewBox="0 0 24 24" fill="none">
                <Circle cx="12" cy="12" r="9" stroke={BrandColors.electricBlue} strokeWidth="2" />
                <Path d="M12 7V12L15 15" stroke={BrandColors.cyanAccent} strokeWidth="2" strokeLinecap="round" />
              </Svg>
            </View>
            <View style={styles.permTextGroup}>
              <Text style={styles.permTitle}>Microphone Permission Required</Text>
              <Text style={styles.permSubtitle}>Access will be requested when you tap Start</Text>
            </View>
          </View>
        );
    }
  };

  return (
    <View style={styles.container}>
      {/* Top Header Bar */}
      <View style={[styles.topBar, { paddingTop: Math.max(insets.top, 24) }]}>
        <Pressable onPress={onBack} hitSlop={12} style={styles.backButton}>
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
        {/* Heading & Subtitle */}
        <Animated.View
          entering={FadeInDown.duration(400)}
          style={styles.headingSection}>
          <Text style={styles.mainHeading}>Record Audio</Text>
          <Text style={styles.descriptionText}>
            Capture a voice sample for authenticity analysis and neural deepfake detection.
          </Text>
        </Animated.View>

        {/* Error Banner if any */}
        {errorMessage && (
          <View style={styles.errorBanner}>
            <Text style={styles.errorText}>{errorMessage}</Text>
          </View>
        )}

        {/* Large Glowing Microphone Illustration with Concentric Rings & Horizontal Waveforms */}
        <Animated.View
          entering={FadeInDown.delay(120).duration(500)}
          style={styles.illustrationWrapper}>
          <Animated.View style={[styles.micIllustrationContainer, animatedMicStyle]}>
            <Svg width={280} height={200} viewBox="0 0 280 200">
              <Defs>
                <SvgLinearGradient id="micOuterRing" x1="0%" y1="0%" x2="100%" y2="100%">
                  <Stop offset="0%" stopColor={BrandColors.primaryPurple} stopOpacity="0.6" />
                  <Stop offset="50%" stopColor={BrandColors.electricBlue} stopOpacity="0.4" />
                  <Stop offset="100%" stopColor={BrandColors.cyanAccent} stopOpacity="0.7" />
                </SvgLinearGradient>
                <SvgLinearGradient id="micGlowBadge" x1="0%" y1="0%" x2="100%" y2="100%">
                  <Stop offset="0%" stopColor="#1E2A5E" />
                  <Stop offset="50%" stopColor="#12183C" />
                  <Stop offset="100%" stopColor="#0B0F2A" />
                </SvgLinearGradient>
                <SvgLinearGradient id="micBodyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <Stop offset="0%" stopColor="#00D9FF" />
                  <Stop offset="50%" stopColor="#168BFF" />
                  <Stop offset="100%" stopColor="#A020F0" />
                </SvgLinearGradient>
                <SvgLinearGradient id="waveBarGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <Stop offset="0%" stopColor={BrandColors.primaryPurple} />
                  <Stop offset="50%" stopColor={BrandColors.electricBlue} />
                  <Stop offset="100%" stopColor={BrandColors.cyanAccent} />
                </SvgLinearGradient>
              </Defs>

              {/* Decorative Horizontal Waveforms Flanking the Microphone */}
              {/* Left Waveform Bars */}
              <Line x1="15" y1="100" x2="80" y2="100" stroke="url(#waveBarGrad)" strokeWidth="1" strokeDasharray="3 3" strokeOpacity="0.4" />
              <Rect x="20" y="93" width="3" height="14" rx="1.5" fill="url(#waveBarGrad)" opacity="0.4" />
              <Rect x="29" y="87" width="3.5" height="26" rx="1.75" fill="url(#waveBarGrad)" opacity="0.6" />
              <Rect x="39" y="80" width="4" height="40" rx="2" fill="url(#waveBarGrad)" opacity="0.8" />
              <Rect x="50" y="72" width="4.5" height="56" rx="2.25" fill="url(#waveBarGrad)" opacity="0.95" />
              <Rect x="62" y="84" width="4" height="32" rx="2" fill="url(#waveBarGrad)" opacity="0.75" />
              <Rect x="73" y="91" width="3.5" height="18" rx="1.75" fill="url(#waveBarGrad)" opacity="0.5" />

              {/* Right Waveform Bars */}
              <Line x1="200" y1="100" x2="265" y2="100" stroke="url(#waveBarGrad)" strokeWidth="1" strokeDasharray="3 3" strokeOpacity="0.4" />
              <Rect x="203" y="91" width="3.5" height="18" rx="1.75" fill="url(#waveBarGrad)" opacity="0.5" />
              <Rect x="214" y="84" width="4" height="32" rx="2" fill="url(#waveBarGrad)" opacity="0.75" />
              <Rect x="225" y="72" width="4.5" height="56" rx="2.25" fill="url(#waveBarGrad)" opacity="0.95" />
              <Rect x="237" y="80" width="4" height="40" rx="2" fill="url(#waveBarGrad)" opacity="0.8" />
              <Rect x="247" y="87" width="3.5" height="26" rx="1.75" fill="url(#waveBarGrad)" opacity="0.6" />
              <Rect x="257" y="93" width="3" height="14" rx="1.5" fill="url(#waveBarGrad)" opacity="0.4" />

              {/* Concentric Neon Target Calibration Rings */}
              <Circle cx="140" cy="100" r="76" stroke="url(#micOuterRing)" strokeWidth="1.2" strokeDasharray="5 7" fill="none" opacity="0.4" />
              <Circle cx="140" cy="100" r="62" stroke={BrandColors.cyanAccent} strokeWidth="1.5" strokeOpacity="0.3" fill="none" />
              <Circle cx="140" cy="100" r="48" stroke={BrandColors.electricBlue} strokeWidth="2" strokeOpacity="0.6" fill="url(#micGlowBadge)" />

              {/* Center Vector Microphone Icon */}
              {/* Mic Capsule Head */}
              <Rect x="133" y="76" width="14" height="26" rx="7" fill="url(#micBodyGrad)" />
              {/* Capsule Mesh Pattern Lines */}
              <Line x1="133" y1="83" x2="147" y2="83" stroke="#FFFFFF" strokeWidth="0.8" strokeOpacity="0.6" />
              <Line x1="133" y1="89" x2="147" y2="89" stroke="#FFFFFF" strokeWidth="0.8" strokeOpacity="0.6" />
              {/* Mic U-Bracket */}
              <Path
                d="M 127 88 C 127 101, 153 101, 153 88"
                stroke="url(#micBodyGrad)"
                strokeWidth="2.5"
                strokeLinecap="round"
                fill="none"
              />
              {/* Mic Stand Post */}
              <Line x1="140" y1="101" x2="140" y2="114" stroke="url(#micBodyGrad)" strokeWidth="2.5" strokeLinecap="round" />
              {/* Mic Stand Base */}
              <Line x1="132" y1="114" x2="148" y2="114" stroke={BrandColors.cyanAccent} strokeWidth="2.5" strokeLinecap="round" />
            </Svg>
          </Animated.View>
        </Animated.View>

        {/* Microphone Permission Status Card */}
        <Animated.View
          entering={FadeInDown.delay(200).duration(450)}
          style={styles.cardContainer}>
          {renderPermissionBadge()}
        </Animated.View>

        {/* Recording Configuration Information Card */}
        <Animated.View
          entering={FadeInDown.delay(280).duration(450)}
          style={styles.configCard}>
          <View style={styles.configHeaderRow}>
            <Text style={styles.configHeading}>Recording Configuration</Text>
            <View style={styles.configBadge}>
              <Text style={styles.configBadgeText}>HIGH FIDELITY</Text>
            </View>
          </View>

          <View style={styles.configRow}>
            <Text style={styles.configLabel}>Audio Format:</Text>
            <Text style={styles.configValue}>AAC (.m4a) High-Quality</Text>
          </View>
          <View style={styles.configRow}>
            <Text style={styles.configLabel}>Sampling Rate:</Text>
            <Text style={styles.configValue}>{DEFAULT_AUDIO_CONFIG.sampleRate / 1000} kHz</Text>
          </View>
          <View style={styles.configRow}>
            <Text style={styles.configLabel}>Audio Channels:</Text>
            <Text style={styles.configValue}>Mono (1-ch Speech Optimized)</Text>
          </View>
          <View style={styles.configRow}>
            <Text style={styles.configLabel}>Privacy Guarantee:</Text>
            <Text style={[styles.configValue, { color: BrandColors.cyanAccent }]}>
              100% Local On-Device
            </Text>
          </View>
        </Animated.View>

        {/* Start Recording Button */}
        <Animated.View
          entering={FadeInDown.delay(360).duration(450)}
          style={[styles.btnWrapper, animatedBtnStyle]}>
          <Pressable
            onPress={onStartRecording}
            onPressIn={() => (startBtnScale.value = withTiming(0.96, { duration: 100 }))}
            onPressOut={() => (startBtnScale.value = withTiming(1, { duration: 150 }))}
            style={styles.startBtnPressable}>
            <LinearGradient
              colors={[BrandColors.primaryPurple, BrandColors.electricBlue, BrandColors.cyanAccent]}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={styles.startBtnGradient}>
              <Svg width={22} height={22} viewBox="0 0 24 24" fill="none">
                <Path
                  d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"
                  fill="#FFFFFF"
                />
                <Path
                  d="M19 10v2a7 7 0 0 1-14 0v-2"
                  stroke="#FFFFFF"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
                <Line x1="12" y1="19" x2="12" y2="23" stroke="#FFFFFF" strokeWidth="2" strokeLinecap="round" />
                <Line x1="8" y1="23" x2="16" y2="23" stroke="#FFFFFF" strokeWidth="2" strokeLinecap="round" />
              </Svg>
              <Text style={styles.startBtnText}>Start Recording</Text>
            </LinearGradient>
          </Pressable>
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
  errorBanner: {
    backgroundColor: 'rgba(239, 68, 68, 0.15)',
    borderWidth: 1,
    borderColor: 'rgba(239, 68, 68, 0.4)',
    borderRadius: 12,
    padding: Spacing.two,
    marginTop: Spacing.two,
  },
  errorText: {
    color: '#EF4444',
    fontSize: 12,
    fontWeight: '600',
    textAlign: 'center',
  },
  illustrationWrapper: {
    alignItems: 'center',
    justifyContent: 'center',
    marginVertical: Spacing.three,
  },
  micIllustrationContainer: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  cardContainer: {
    marginBottom: Spacing.three,
  },
  permCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    padding: 14,
    borderRadius: 14,
    borderWidth: 1,
  },
  permGranted: {
    backgroundColor: 'rgba(0, 217, 255, 0.08)',
    borderColor: 'rgba(0, 217, 255, 0.3)',
  },
  permDenied: {
    backgroundColor: 'rgba(239, 68, 68, 0.08)',
    borderColor: 'rgba(239, 68, 68, 0.3)',
  },
  permPending: {
    backgroundColor: 'rgba(22, 139, 255, 0.08)',
    borderColor: 'rgba(22, 139, 255, 0.25)',
  },
  permIconBadge: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: 'rgba(0, 217, 255, 0.12)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  permTextGroup: {
    flex: 1,
  },
  permTitle: {
    color: '#FFFFFF',
    fontSize: 13.5,
    fontWeight: '700',
  },
  permSubtitle: {
    color: BrandColors.secondaryText,
    fontSize: 11.5,
    fontWeight: '500',
    marginTop: 2,
  },
  configCard: {
    backgroundColor: BrandColors.cardBackground,
    borderRadius: 16,
    padding: Spacing.three,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    marginBottom: Spacing.four,
  },
  configHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: Spacing.two,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.06)',
    paddingBottom: Spacing.two,
  },
  configHeading: {
    color: '#FFFFFF',
    fontSize: 13.5,
    fontWeight: '700',
  },
  configBadge: {
    backgroundColor: 'rgba(22, 139, 255, 0.15)',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 8,
  },
  configBadgeText: {
    color: BrandColors.cyanAccent,
    fontSize: 9.5,
    fontWeight: '700',
    letterSpacing: 0.8,
  },
  configRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 5,
  },
  configLabel: {
    color: BrandColors.secondaryText,
    fontSize: 12.5,
    fontWeight: '500',
  },
  configValue: {
    color: '#FFFFFF',
    fontSize: 12.5,
    fontWeight: '600',
  },
  btnWrapper: {
    width: '100%',
  },
  startBtnPressable: {
    borderRadius: 16,
    overflow: 'hidden',
  },
  startBtnGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    gap: 10,
  },
  startBtnText: {
    color: '#FFFFFF',
    fontSize: 16.5,
    fontWeight: '800',
    letterSpacing: 0.6,
  },
});

