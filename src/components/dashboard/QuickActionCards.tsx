import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import Animated, {
  Easing,
  FadeInDown,
  useAnimatedStyle,
  useSharedValue,
  withTiming,
} from 'react-native-reanimated';
import Svg, {
  Circle,
  Defs,
  Line,
  LinearGradient as SvgLinearGradient,
  Path,
  Rect,
  Stop,
} from 'react-native-svg';

import { BrandColors, Spacing } from '@/constants/theme';

interface QuickActionCardsProps {
  onRecordPress: () => void;
  onUploadPress: () => void;
}

export const QuickActionCards: React.FC<QuickActionCardsProps> = ({
  onRecordPress,
  onUploadPress,
}) => {
  const recordScale = useSharedValue(1);
  const uploadScale = useSharedValue(1);

  const animatedRecordStyle = useAnimatedStyle(() => ({
    transform: [{ scale: recordScale.value }],
  }));

  const animatedUploadStyle = useAnimatedStyle(() => ({
    transform: [{ scale: uploadScale.value }],
  }));

  return (
    <Animated.View
      entering={FadeInDown.delay(280).duration(480).easing(Easing.out(Easing.cubic))}
      style={styles.container}>
      {/* ============================================================ */}
      {/* CARD 1: RECORD AUDIO                                         */}
      {/* ============================================================ */}
      <Animated.View style={[styles.cardWrapper, animatedRecordStyle]}>
        <Pressable
          onPress={onRecordPress}
          onPressIn={() => (recordScale.value = withTiming(0.96, { duration: 120 }))}
          onPressOut={() => (recordScale.value = withTiming(1, { duration: 180 }))}
          style={styles.cardPressable}>
          <View style={styles.cardContent}>
            {/* Glowing Microphone Icon */}
            <View style={[styles.iconCircle, styles.recordGlow]}>
              <Svg width={26} height={26} viewBox="0 0 24 24" fill="none">
                <Defs>
                  <SvgLinearGradient id="micCardGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <Stop offset="0%" stopColor="#00D9FF" />
                    <Stop offset="100%" stopColor={BrandColors.electricBlue} />
                  </SvgLinearGradient>
                </Defs>
                {/* Capsule */}
                <Rect x="8" y="2" width="8" height="12" rx="4" fill="url(#micCardGrad)" />
                {/* U-Bracket */}
                <Path
                  d="M 5 10 C 5 14.5, 8 18, 12 18 C 16 18, 19 14.5, 19 10"
                  stroke="#00D9FF"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
                {/* Stem & Base */}
                <Line x1="12" y1="18" x2="12" y2="22" stroke="#00D9FF" strokeWidth="2" strokeLinecap="round" />
                <Line x1="8" y1="22" x2="16" y2="22" stroke="#00D9FF" strokeWidth="2" strokeLinecap="round" />
              </Svg>
            </View>

            {/* Title & Description */}
            <Text style={styles.cardTitle}>Record Audio</Text>
            <Text style={styles.cardDescription}>Capture and analyze in real time</Text>
          </View>
        </Pressable>
      </Animated.View>

      {/* ============================================================ */}
      {/* CARD 2: UPLOAD AUDIO                                         */}
      {/* ============================================================ */}
      <Animated.View style={[styles.cardWrapper, animatedUploadStyle]}>
        <Pressable
          onPress={onUploadPress}
          onPressIn={() => (uploadScale.value = withTiming(0.96, { duration: 120 }))}
          onPressOut={() => (uploadScale.value = withTiming(1, { duration: 180 }))}
          style={styles.cardPressable}>
          <View style={styles.cardContent}>
            {/* Glowing Cloud/Upload Icon */}
            <View style={[styles.iconCircle, styles.uploadGlow]}>
              <Svg width={26} height={26} viewBox="0 0 24 24" fill="none">
                <Defs>
                  <SvgLinearGradient id="uploadCardGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <Stop offset="0%" stopColor="#C040FF" />
                    <Stop offset="100%" stopColor={BrandColors.primaryPurple} />
                  </SvgLinearGradient>
                </Defs>
                {/* Cloud Outline */}
                <Path
                  d="M 6.5 18 C 4 18, 2 16, 2 13.5 C 2 11.2, 3.6 9.3, 5.8 9.1 C 6.5 5.8, 9.4 3.5, 12.8 3.5 C 16.5 3.5, 19.6 6.3, 20 10 C 21.8 10.4, 23 12, 23 13.8 C 23 16.1, 21.1 18, 18.8 18 L 17 18"
                  stroke="#C040FF"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                {/* Upload Arrow */}
                <Path
                  d="M 12 11 L 12 20 M 8.5 14.5 L 12 11 L 15.5 14.5"
                  stroke="#FFFFFF"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </Svg>
            </View>

            {/* Title & Description */}
            <Text style={styles.cardTitle}>Upload Audio</Text>
            <Text style={styles.cardDescription}>Analyze an existing audio file</Text>
          </View>
        </Pressable>
      </Animated.View>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '100%',
    flexDirection: 'row',
    paddingHorizontal: Spacing.four,
    gap: Spacing.three,
    marginBottom: Spacing.four,
  },
  cardWrapper: {
    flex: 1,
  },
  cardPressable: {
    backgroundColor: '#141A36',
    borderRadius: 18,
    borderWidth: 1.25,
    borderColor: 'rgba(22, 139, 255, 0.22)',
    overflow: 'hidden',
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 4,
  },
  cardContent: {
    padding: Spacing.three,
    minHeight: 140,
    justifyContent: 'space-between',
  },
  iconCircle: {
    width: 44,
    height: 44,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.two,
  },
  recordGlow: {
    backgroundColor: 'rgba(0, 217, 255, 0.1)',
    borderWidth: 1,
    borderColor: 'rgba(0, 217, 255, 0.35)',
  },
  uploadGlow: {
    backgroundColor: 'rgba(160, 32, 240, 0.12)',
    borderWidth: 1,
    borderColor: 'rgba(192, 64, 255, 0.35)',
  },
  cardTitle: {
    color: '#FFFFFF',
    fontSize: 15.5,
    fontWeight: '700',
    letterSpacing: 0.2,
    marginBottom: 4,
  },
  cardDescription: {
    color: BrandColors.secondaryText,
    fontSize: 11.5,
    lineHeight: 16,
    fontWeight: '500',
  },
});

export default QuickActionCards;

