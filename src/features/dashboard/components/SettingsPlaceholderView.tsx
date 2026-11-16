import { LinearGradient } from 'expo-linear-gradient';
import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import Animated, { Easing, FadeInDown } from 'react-native-reanimated';
import Svg, { Circle, Defs, LinearGradient as SvgLinearGradient, Path, Stop } from 'react-native-svg';

import { BrandColors, Spacing } from '@/constants/theme';

interface SettingsPlaceholderViewProps {
  onGoHome?: () => void;
}

export const SettingsPlaceholderView: React.FC<SettingsPlaceholderViewProps> = ({
  onGoHome,
}) => {
  return (
    <Animated.View
      entering={FadeInDown.duration(400).easing(Easing.out(Easing.cubic))}
      style={styles.container}>
      {/* Category Badge */}
      <View style={styles.badge}>
        <View style={styles.badgeDot} />
        <Text style={styles.badgeText}>CONFIGURATION PREVIEW</Text>
      </View>

      {/* Vector Illustration */}
      <View style={styles.iconCircle}>
        <Svg width={64} height={64} viewBox="0 0 64 64" fill="none">
          <Defs>
            <SvgLinearGradient id="settingsGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <Stop offset="0%" stopColor="#00D9FF" />
              <Stop offset="100%" stopColor="#C040FF" />
            </SvgLinearGradient>
          </Defs>
          <Circle cx="32" cy="32" r="9" stroke="url(#settingsGrad)" strokeWidth="3.5" fill="#141A36" />
          <Path
            d="M 51.5 40 A 4.4 4.4 0 0 0 52.3 44.8 L 52.6 45.1 A 5.3 5.3 0 1 1 45.1 52.6 L 44.8 52.3 A 4.4 4.4 0 0 0 40 51.5 A 4.4 4.4 0 0 0 37 55.5 L 37 56 A 5.3 5.3 0 1 1 26.4 56 L 26.4 55.5 A 4.4 4.4 0 0 0 23.5 51.5 A 4.4 4.4 0 0 0 18.7 52.3 L 18.4 52.6 A 5.3 5.3 0 1 1 10.9 45.1 L 11.2 44.8 A 4.4 4.4 0 0 0 12 40 A 4.4 4.4 0 0 0 8 37 L 7.5 37 A 5.3 5.3 0 1 1 7.5 26.4 L 8 26.4 A 4.4 4.4 0 0 0 12 23.5 A 4.4 4.4 0 0 0 11.2 18.7 L 10.9 18.4 A 5.3 5.3 0 1 1 18.4 10.9 L 18.7 11.2 A 4.4 4.4 0 0 0 23.5 12 A 4.4 4.4 0 0 0 26.4 8 L 26.4 7.5 A 5.3 5.3 0 1 1 37 7.5 L 37 8 A 4.4 4.4 0 0 0 40 12 A 4.4 4.4 0 0 0 44.8 11.2 L 45.1 10.9 A 5.3 5.3 0 1 1 52.6 18.4 L 52.3 18.7 A 4.4 4.4 0 0 0 51.5 23.5 A 4.4 4.4 0 0 0 55.5 26.4 L 56 26.4 A 5.3 5.3 0 1 1 56 37 L 55.5 37 A 4.4 4.4 0 0 0 51.5 40 Z"
            stroke="url(#settingsGrad)"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </Svg>
      </View>

      {/* Title & Description */}
      <Text style={styles.titleText}>Settings & Security</Text>
      <Text style={styles.descriptionText}>
        AI model detection sensitivity thresholds, automated background screening, and privacy preferences will be configurable here.
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

export default SettingsPlaceholderView;

