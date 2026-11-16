import { LinearGradient } from 'expo-linear-gradient';
import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import Animated, { Easing, FadeInDown } from 'react-native-reanimated';
import Svg, { Circle, Defs, LinearGradient as SvgLinearGradient, Path, Stop } from 'react-native-svg';

import { BrandColors, Spacing } from '@/constants/theme';

interface HistoryPlaceholderViewProps {
  onGoHome?: () => void;
}

export const HistoryPlaceholderView: React.FC<HistoryPlaceholderViewProps> = ({
  onGoHome,
}) => {
  return (
    <Animated.View
      entering={FadeInDown.duration(400).easing(Easing.out(Easing.cubic))}
      style={styles.container}>
      {/* Category Badge */}
      <View style={styles.badge}>
        <View style={styles.badgeDot} />
        <Text style={styles.badgeText}>PHASE 4 PREVIEW</Text>
      </View>

      {/* Vector Illustration */}
      <View style={styles.iconCircle}>
        <Svg width={64} height={64} viewBox="0 0 64 64" fill="none">
          <Defs>
            <SvgLinearGradient id="historyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <Stop offset="0%" stopColor="#00D9FF" />
              <Stop offset="100%" stopColor={BrandColors.electricBlue} />
            </SvgLinearGradient>
          </Defs>
          <Circle cx="32" cy="32" r="26" stroke="url(#historyGrad)" strokeWidth="3" fill="#141A36" />
          <Path
            d="M 32 18 L 32 32 L 42 38"
            stroke="#FFFFFF"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </Svg>
      </View>

      {/* Title & Description */}
      <Text style={styles.titleText}>Analysis History</Text>
      <Text style={styles.descriptionText}>
        Past voice verification logs, risk assessment scores, and exportable authenticity reports will be stored and organized here.
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
    backgroundColor: 'rgba(22, 139, 255, 0.1)',
    borderWidth: 1,
    borderColor: 'rgba(0, 217, 255, 0.25)',
  },
  badgeDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: BrandColors.cyanAccent,
  },
  badgeText: {
    color: BrandColors.cyanAccent,
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
    borderColor: 'rgba(22, 139, 255, 0.3)',
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

export default HistoryPlaceholderView;

