import { LinearGradient } from 'expo-linear-gradient';
import * as SplashScreen from 'expo-splash-screen';
import { StatusBar } from 'expo-status-bar';
import React, { useEffect, useRef, useState } from 'react';
import {
  Dimensions,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import Animated, {
  Easing,
  FadeIn,
  FadeInDown,
  runOnJS,
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withTiming,
} from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Svg, {
  Circle,
  Defs,
  LinearGradient as SvgLinearGradient,
  RadialGradient,
  Stop,
  Text as SvgText,
  TSpan,
} from 'react-native-svg';

import { FlowingWaveform } from '@/components/animations/FlowingWaveform';
import { VoiceShieldLogo } from '@/components/branding/VoiceShieldLogo';
import { BrandColors, Spacing } from '@/constants/theme';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

export interface VoiceShieldSplashScreenProps {
  /** If true, remains on screen as a standalone preview screen; if false, acts as app launch overlay */
  standalone?: boolean;
  /** Duration in milliseconds the splash remains visible after React is ready (default 3000ms / 3s) */
  displayDuration?: number;
  /** Duration in milliseconds of the exit fade-out animation (default 500ms) */
  fadeDuration?: number;
  /** Callback fired when splash sequence completes */
  onFinish?: () => void;
}

export const VoiceShieldSplashScreen: React.FC<VoiceShieldSplashScreenProps> = ({
  standalone = false,
  displayDuration = 3000,
  fadeDuration = 500,
  onFinish,
}) => {
  const insets = useSafeAreaInsets();
  const [isVisible, setIsVisible] = useState(true);
  const [isFadingOut, setIsFadingOut] = useState(false);

  // References to ensure clean lifecycle, prevent repeated timers, and avoid memory leaks
  const isMountedRef = useRef(true);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hasStartedRef = useRef(false);

  // Reanimated values for entrance & exit animations
  const logoScale = useSharedValue(0.85);
  const logoOpacity = useSharedValue(0);
  const radialGlowScale = useSharedValue(0.7);
  const radialGlowOpacity = useSharedValue(0);
  const containerOpacity = useSharedValue(1);

  const handleFinish = () => {
    if (!isMountedRef.current) return;
    setIsVisible(false);
    onFinish?.();
  };

  useEffect(() => {
    isMountedRef.current = true;

    // Prevent re-triggering if parent re-renders
    if (hasStartedRef.current) return;
    hasStartedRef.current = true;

    async function initSplashLifecycle() {
      // 1. Immediately dismiss the native OS splash screen without artificial delay
      //    so our custom React Native UI is revealed and interactive immediately.
      try {
        await SplashScreen.hideAsync();
      } catch {
        // Safe fallback if already hidden or unsupported platform
      }

      if (!isMountedRef.current) return;

      // 2. Animate logo entrance smoothly
      logoOpacity.value = withTiming(1, {
        duration: 550,
        easing: Easing.out(Easing.cubic),
      });

      logoScale.value = withSpring(1, {
        damping: 14,
        stiffness: 90,
        mass: 0.8,
      });

      // 3. Radial ambient glow expansion behind the shield
      radialGlowOpacity.value = withTiming(0.9, { duration: 750 });
      radialGlowScale.value = withTiming(1, {
        duration: 1100,
        easing: Easing.out(Easing.ease),
      });

      // If standalone preview mode, do not auto-dismiss
      if (standalone) return;

      // 4. Guaranteed hold: keep the splash screen visible for at least 3 seconds
      //    (displayDuration) AFTER the React application is ready.
      timerRef.current = setTimeout(() => {
        if (!isMountedRef.current) return;

        // Set pointerEvents="none" so user can immediately touch underlying app as fade begins
        setIsFadingOut(true);

        // 5. Smooth 500ms fade out transition
        containerOpacity.value = withTiming(
          0,
          {
            duration: fadeDuration,
            easing: Easing.inOut(Easing.ease),
          },
          (finished) => {
            if (finished) {
              runOnJS(handleFinish)();
            }
          }
        );
      }, displayDuration);
    }

    initSplashLifecycle();

    return () => {
      isMountedRef.current = false;
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [standalone, displayDuration, fadeDuration]);

  const animatedLogoStyle = useAnimatedStyle(() => ({
    opacity: logoOpacity.value,
    transform: [{ scale: logoScale.value }],
  }));

  const animatedRadialStyle = useAnimatedStyle(() => ({
    opacity: radialGlowOpacity.value,
    transform: [{ scale: radialGlowScale.value }],
  }));

  const animatedContainerStyle = useAnimatedStyle(() => ({
    opacity: containerOpacity.value,
  }));

  // Once faded out completely, unmount completely from React tree
  // ensuring zero interference with Expo Router navigation or layout
  if (!isVisible && !standalone) {
    return null;
  }

  // Dynamic logo sizing tailored for wide-span side wings (1.4:1 aspect ratio)
  const logoHeight = Math.min(SCREEN_WIDTH * 0.5, SCREEN_HEIGHT * 0.25, 195);
  const logoWidth = Math.min(SCREEN_WIDTH * 0.94, Math.round(logoHeight * 1.4));

  // Responsive wordmark metrics with balanced 10-12% font reduction & ample headroom
  const wordmarkWidth = Math.min(SCREEN_WIDTH * 0.88, 340);
  const wordmarkFontSize = Math.min(30, Math.max(23, Math.round(SCREEN_WIDTH * 0.073)));

  return (
    <Animated.View
      pointerEvents={isFadingOut ? 'none' : 'auto'}
      style={[
        StyleSheet.absoluteFill,
        styles.rootContainer,
        !standalone && animatedContainerStyle,
      ]}>
      <StatusBar style="light" />

      {/* Deep Navy/Black Gradient Background */}
      <LinearGradient
        colors={[BrandColors.primaryBackground, BrandColors.secondaryBackground, '#050713']}
        start={{ x: 0.5, y: 0 }}
        end={{ x: 0.5, y: 1 }}
        style={StyleSheet.absoluteFill}
      />

      {/* Background Radial Purple-Blue Ambient Glow behind Logo */}
      <Animated.View
        pointerEvents="none"
        style={[
          styles.radialGlowWrapper,
          { top: SCREEN_HEIGHT * 0.22 },
          animatedRadialStyle,
        ]}>
        <Svg width={SCREEN_WIDTH} height={SCREEN_WIDTH} viewBox="0 0 360 360">
          <Defs>
            <RadialGradient id="bgCenterGlow" cx="50%" cy="50%" r="50%">
              <Stop offset="0%" stopColor={BrandColors.electricBlue} stopOpacity="0.32" />
              <Stop offset="35%" stopColor={BrandColors.primaryPurple} stopOpacity="0.22" />
              <Stop offset="70%" stopColor={BrandColors.cyanAccent} stopOpacity="0.06" />
              <Stop offset="100%" stopColor="#070817" stopOpacity="0" />
            </RadialGradient>
          </Defs>
          <Circle cx="180" cy="180" r="180" fill="url(#bgCenterGlow)" />
        </Svg>
      </Animated.View>

      {/* Content Container (Respecting Safe Area & Android notches) */}
      <View
        style={[
          styles.contentContainer,
          {
            paddingTop: Math.max(insets.top, 28) + Spacing.three,
            paddingBottom: Math.max(insets.bottom, 16) + 105,
          },
        ]}>
        {/* Top Spacer */}
        <View style={styles.topSpacer} />

        {/* Center Section: Logo + Typography */}
        <View style={styles.centerSection}>
          {/* Logo with Reanimated Entrance & Continuous Breathing Glow */}
          <Animated.View style={animatedLogoStyle}>
            <VoiceShieldLogo
              width={logoWidth}
              height={logoHeight}
              animated={true}
              expanded={true}
            />
          </Animated.View>

          {/* Brand Name Typography with Consistent Electric-Blue to Cyan Gradient */}
          <Animated.View
            entering={FadeInDown.delay(200).duration(500).easing(Easing.out(Easing.cubic))}
            style={styles.brandTitleContainer}>
            <Svg
              height={48}
              width={wordmarkWidth}
              viewBox="0 0 340 48"
              style={{ overflow: 'visible' }}>
              <Defs>
                <SvgLinearGradient id="brandTitleGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <Stop offset="0%" stopColor="#3AA0FF" />
                  <Stop offset="35%" stopColor={BrandColors.electricBlue} />
                  <Stop offset="75%" stopColor={BrandColors.cyanAccent} />
                  <Stop offset="100%" stopColor="#7AEFFF" />
                </SvgLinearGradient>
              </Defs>
              <SvgText
                x="170"
                y="34"
                textAnchor="middle"
                fontSize={wordmarkFontSize}
                fontWeight="800"
                letterSpacing="0.8"
                fill="url(#brandTitleGrad)">
                VoiceShield
              </SvgText>
            </Svg>
          </Animated.View>

          {/* Subtitle / Tagline */}
          <Animated.View
            entering={FadeInDown.delay(350).duration(500).easing(Easing.out(Easing.cubic))}
            style={styles.taglineContainer}>
            <Text style={styles.taglineText}>
              AI-POWERED VOICE AUTHENTICITY DETECTION
            </Text>

            {/* Subtle Cybersecurity Verification Dot */}
            <View style={styles.securityBadge}>
              <View style={styles.securityPulseDot} />
              <Text style={styles.securityBadgeText}>ENCRYPTED NEURAL ENGINE</Text>
            </View>
          </Animated.View>
        </View>

        {/* Bottom Section Spacer */}
        <View style={styles.bottomSpacer} />
      </View>

      {/* Bottom Section: Flowing Purple-Blue Audio Waveform Positioned Above System Navigation Bar */}
      <Animated.View
        entering={FadeIn.delay(450).duration(600)}
        style={[
          styles.bottomWaveformWrapper,
          { bottom: Math.max(insets.bottom, 0) },
        ]}>
        <FlowingWaveform height={100} />
      </Animated.View>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  rootContainer: {
    flex: 1,
    backgroundColor: BrandColors.primaryBackground,
    zIndex: 9999,
  },
  radialGlowWrapper: {
    position: 'absolute',
    left: 0,
    right: 0,
    alignItems: 'center',
    justifyContent: 'center',
  },
  contentContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: Spacing.four,
  },
  topSpacer: {
    flex: 0.7,
  },
  centerSection: {
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
  },
  brandTitleContainer: {
    marginTop: Spacing.three,
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
  },
  taglineContainer: {
    marginTop: Spacing.two,
    alignItems: 'center',
    gap: Spacing.two,
    paddingHorizontal: Spacing.three,
  },
  taglineText: {
    color: BrandColors.secondaryText,
    fontSize: 11.5,
    fontWeight: '600',
    letterSpacing: 1.5,
    textAlign: 'center',
    textTransform: 'uppercase',
  },
  securityBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 3,
    borderRadius: 12,
    backgroundColor: 'rgba(22, 139, 255, 0.08)',
    borderWidth: 1,
    borderColor: 'rgba(0, 217, 255, 0.2)',
    marginTop: 4,
  },
  securityPulseDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: BrandColors.cyanAccent,
  },
  securityBadgeText: {
    color: BrandColors.cyanAccent,
    fontSize: 9.5,
    fontWeight: '700',
    letterSpacing: 1.2,
  },
  bottomSpacer: {
    flex: 1.2,
  },
  bottomWaveformWrapper: {
    position: 'absolute',
    left: 0,
    right: 0,
    height: 100,
    pointerEvents: 'none',
  },
});

export default VoiceShieldSplashScreen;
