import { LinearGradient } from 'expo-linear-gradient';
import React, { useEffect } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import Animated, {
  Easing,
  FadeInDown,
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withSequence,
  withTiming,
} from 'react-native-reanimated';
import Svg, {
  Circle,
  Defs,
  LinearGradient as SvgLinearGradient,
  Path,
  Rect,
  Stop,
} from 'react-native-svg';

import { BrandColors, Spacing } from '@/constants/theme';

interface MainAuthenticityCardProps {
  onPress: () => void;
}

export const MainAuthenticityCard: React.FC<MainAuthenticityCardProps> = ({
  onPress,
}) => {
  // Waveform gentle pulsing shared values
  const pulse1 = useSharedValue(1);
  const pulse2 = useSharedValue(0.7);
  const cardScale = useSharedValue(1);

  useEffect(() => {
    pulse1.value = withRepeat(
      withSequence(
        withTiming(1.3, { duration: 1300, easing: Easing.inOut(Easing.cubic) }),
        withTiming(0.8, { duration: 1300, easing: Easing.inOut(Easing.cubic) })
      ),
      -1,
      true
    );

    pulse2.value = withRepeat(
      withSequence(
        withTiming(0.75, { duration: 1600, easing: Easing.inOut(Easing.cubic) }),
        withTiming(1.25, { duration: 1600, easing: Easing.inOut(Easing.cubic) })
      ),
      -1,
      true
    );
  }, [pulse1, pulse2]);

  const animatedWaveBar1 = useAnimatedStyle(() => ({
    transform: [{ scaleY: pulse1.value }],
  }));

  const animatedWaveBar2 = useAnimatedStyle(() => ({
    transform: [{ scaleY: pulse2.value }],
  }));

  const animatedCardStyle = useAnimatedStyle(() => ({
    transform: [{ scale: cardScale.value }],
  }));

  const handlePressIn = () => {
    cardScale.value = withTiming(0.98, { duration: 120 });
  };

  const handlePressOut = () => {
    cardScale.value = withTiming(1.0, { duration: 180 });
  };

  return (
    <Animated.View
      entering={FadeInDown.delay(200).duration(500).easing(Easing.out(Easing.cubic))}
      style={[styles.containerWrapper, animatedCardStyle]}>
      <Pressable
        onPress={onPress}
        onPressIn={handlePressIn}
        onPressOut={handlePressOut}
        style={styles.pressable}>
        <LinearGradient
          colors={['#7E17C2', '#168BFF', '#0D1A4A']}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.gradientCard}>
          {/* Subtle Ambient Shield Background Decal */}
          <View style={styles.decalWrapper} pointerEvents="none">
            <Svg width={120} height={120} viewBox="0 0 120 120">
              <Path
                d="M 60 15 C 72 15, 90 20, 96 25 C 98 52, 92 85, 60 108 C 28 85, 22 52, 24 25 C 30 20, 48 15, 60 15 Z"
                fill="none"
                stroke="rgba(255, 255, 255, 0.08)"
                strokeWidth="1.5"
              />
            </Svg>
          </View>

          {/* Left Icon: Glowing Audio Waveform Icon */}
          <View style={styles.iconCircle}>
            <Svg width={28} height={28} viewBox="0 0 28 28">
              <Defs>
                <SvgLinearGradient id="waveIconGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <Stop offset="0%" stopColor="#00D9FF" />
                  <Stop offset="100%" stopColor="#FFFFFF" />
                </SvgLinearGradient>
              </Defs>
              <Rect x="2" y="10" width="3" height="8" rx="1.5" fill="url(#waveIconGrad)" />
              <Rect x="7" y="6" width="3" height="16" rx="1.5" fill="url(#waveIconGrad)" />
              <Rect x="12" y="3" width="3.5" height="22" rx="1.75" fill="url(#waveIconGrad)" />
              <Rect x="18" y="7" width="3" height="14" rx="1.5" fill="url(#waveIconGrad)" />
              <Rect x="23" y="11" width="3" height="6" rx="1.5" fill="url(#waveIconGrad)" />
            </Svg>
          </View>

          {/* Center: Title & Subtitle */}
          <View style={styles.textColumn}>
            <View style={styles.badgeRow}>
              <View style={styles.badgeDot} />
              <Text style={styles.badgeLabel}>CORE SCANNER</Text>
            </View>
            <Text style={styles.titleText}>Check Voice Authenticity</Text>
            <Text style={styles.subtitleText}>Detect real or AI-generated voices</Text>
          </View>

          {/* Right: Circular Right-Facing Arrow */}
          <View style={styles.arrowCircle}>
            <Svg width={18} height={18} viewBox="0 0 24 24" fill="none">
              <Path
                d="M 9 6 L 15 12 L 9 18"
                stroke="#FFFFFF"
                strokeWidth={2.5}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </Svg>
          </View>
        </LinearGradient>
      </Pressable>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  containerWrapper: {
    width: '100%',
    paddingHorizontal: Spacing.four,
    marginBottom: Spacing.four,
  },
  pressable: {
    borderRadius: 22,
    overflow: 'hidden',
    shadowColor: BrandColors.electricBlue,
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.35,
    shadowRadius: 14,
    elevation: 8,
  },
  gradientCard: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: Spacing.four,
    paddingHorizontal: Spacing.four,
    borderRadius: 22,
    borderWidth: 1.25,
    borderColor: 'rgba(0, 217, 255, 0.45)',
    position: 'relative',
    overflow: 'hidden',
  },
  decalWrapper: {
    position: 'absolute',
    right: -10,
    bottom: -20,
    opacity: 0.8,
  },
  iconCircle: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: 'rgba(7, 8, 23, 0.45)',
    borderWidth: 1,
    borderColor: 'rgba(0, 217, 255, 0.4)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: Spacing.three,
  },
  textColumn: {
    flex: 1,
    gap: 3,
  },
  badgeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    marginBottom: 2,
  },
  badgeDot: {
    width: 5,
    height: 5,
    borderRadius: 2.5,
    backgroundColor: BrandColors.cyanAccent,
  },
  badgeLabel: {
    color: BrandColors.cyanAccent,
    fontSize: 9.5,
    fontWeight: '800',
    letterSpacing: 1.2,
  },
  titleText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '800',
    letterSpacing: 0.3,
  },
  subtitleText: {
    color: '#E0E7FF',
    fontSize: 12.5,
    fontWeight: '500',
    letterSpacing: 0.2,
  },
  arrowCircle: {
    width: 34,
    height: 34,
    borderRadius: 17,
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: Spacing.two,
  },
});

export default MainAuthenticityCard;

