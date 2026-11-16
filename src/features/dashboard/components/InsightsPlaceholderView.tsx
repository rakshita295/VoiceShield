import { LinearGradient } from 'expo-linear-gradient';
import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import Animated, { Easing, FadeInDown } from 'react-native-reanimated';
import Svg, { Defs, LinearGradient as SvgLinearGradient, Path, Rect, Stop } from 'react-native-svg';

import { BrandColors, Spacing } from '@/constants/theme';

interface InsightsPlaceholderViewProps {
  onGoHome?: () => void;
}

export const InsightsPlaceholderView: React.FC<InsightsPlaceholderViewProps> = ({
  onGoHome,
}) => {
  return (
    <Animated.View
      entering={FadeInDown.duration(400).easing(Easing.out(Easing.cubic))}
      style={styles.container}>
      {/* Category Badge */}
      <View style={styles.badge}>
        <View style={styles.badgeDot} />
        <Text style={styles.badgeText}>ANALYTICS PREVIEW</Text>
      </View>

      {/* Vector Illustration */}
      <View style={styles.iconCircle}>
        <Svg width={64} height={64} viewBox="0 0 64 64" fill="none">
          <Defs>
            <SvgLinearGradient id="insightsGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <Stop offset="0%" stopColor="#C040FF" />
              <Stop offset="100%" stopColor={BrandColors.cyanAccent} />
            </SvgLinearGradient>
          </Defs>
          <Rect x="10" y="36" width="9" height="18" rx="3" fill="#168BFF" />
          <Rect x="23" y="24" width="9" height="30" rx="3" fill="#00D9FF" />
          <Rect x="36" y="14" width="9" height="40" rx="3" fill="#C040FF" />
          <Rect x="49" y="28" width="9" height="26" rx="3" fill="#A020F0" />
          <Path
            d="M 12 30 L 26 18 L 40 8 L 54 22"
            stroke="#FFFFFF"
            strokeWidth="2.5"
            strokeLinecap="round"
          />
        </Svg>
      </View>

      {/* Title & Description */}
      <Text style={styles.titleText}>Deepfake Insights</Text>
      <Text style={styles.descriptionText}>
        Synthetic voice frequency breakdowns, scam pattern detection trends, and biometric confidence charts will be visualized here.
      </Text>

      {/* Return to Home Action */}
      {onGoHome && (
        <Pressable
          onPress={onGoHome}
          style={({ pressed }) => [
            styles.buttonWrapper,
            pressed && styles.buttonPressed,
          ]}>
          <LinearGradient
            colors={[BrandColors.primaryPurple, BrandColors.electricBlue]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={styles.buttonGradient}>
            <Text style={styles.buttonText}>Return to Home</Text>
          </LinearGradient>
        </Pressable>
      )}
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: Spacing.four,
    gap: Spacing.three,
  },
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
    backgroundColor: 'rgba(160, 32, 240, 0.12)',
    borderWidth: 1,
    borderColor: 'rgba(192, 64, 255, 0.25)',
  },
  badgeDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#C040FF',
  },
  badgeText: {
    color: '#C040FF',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 1.2,
  },
  iconCircle: {
    width: 90,
    height: 90,
    borderRadius: 45,
    backgroundColor: 'rgba(13, 18, 48, 0.8)',
    borderWidth: 1.25,
    borderColor: 'rgba(192, 64, 255, 0.3)',
    alignItems: 'center',
    justifyContent: 'center',
    marginVertical: Spacing.two,
  },
  titleText: {
    color: '#FFFFFF',
    fontSize: 24,
    fontWeight: '800',
    textAlign: 'center',
    letterSpacing: 0.3,
  },
  descriptionText: {
    color: BrandColors.secondaryText,
    fontSize: 14,
    lineHeight: 22,
    textAlign: 'center',
    maxWidth: 300,
  },
  buttonWrapper: {
    marginTop: Spacing.three,
    borderRadius: 24,
    overflow: 'hidden',
  },
  buttonPressed: {
    opacity: 0.85,
  },
  buttonGradient: {
    paddingVertical: 12,
    paddingHorizontal: Spacing.five,
    borderRadius: 24,
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 14.5,
    fontWeight: '700',
    letterSpacing: 0.3,
  },
});

export default InsightsPlaceholderView;

