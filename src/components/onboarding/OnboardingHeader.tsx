import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import Svg, {
  Defs,
  LinearGradient as SvgLinearGradient,
  Stop,
  Text as SvgText,
  TSpan,
} from 'react-native-svg';

import { VoiceShieldLogo } from '@/components/branding/VoiceShieldLogo';
import { BrandColors, Spacing } from '@/constants/theme';

interface OnboardingHeaderProps {
  onSkip: () => void;
  showSkip?: boolean;
}

export const OnboardingHeader: React.FC<OnboardingHeaderProps> = ({
  onSkip,
  showSkip = true,
}) => {
  return (
    <View style={styles.headerContainer}>
      {/* Brand Identity (Logo + Wordmark) */}
      <View style={styles.brandRow}>
        <VoiceShieldLogo size={34} animated={false} />
        <View style={styles.wordmarkContainer}>
          <Svg height={26} width={130} viewBox="0 0 130 26">
            <Defs>
              <SvgLinearGradient id="headerBrandGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <Stop offset="0%" stopColor={BrandColors.electricBlue} />
                <Stop offset="100%" stopColor={BrandColors.cyanAccent} />
              </SvgLinearGradient>
            </Defs>
            <SvgText
              x="0"
              y="20"
              fontSize="20"
              fontWeight="800"
              letterSpacing="0.5">
              <TSpan fill="#FFFFFF">Voice</TSpan>
              <TSpan fill="url(#headerBrandGrad)">Shield</TSpan>
            </SvgText>
          </Svg>
        </View>
      </View>

      {/* Skip Button */}
      {showSkip ? (
        <Pressable
          onPress={onSkip}
          hitSlop={{ top: 12, bottom: 12, left: 16, right: 16 }}
          style={({ pressed }) => [
            styles.skipButton,
            pressed && styles.skipButtonPressed,
          ]}>
          <Text style={styles.skipText}>Skip</Text>
        </Pressable>
      ) : (
        <View style={styles.skipPlaceholder} />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  headerContainer: {
    width: '100%',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: Spacing.four,
    paddingVertical: Spacing.two,
  },
  brandRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.two,
  },
  wordmarkContainer: {
    justifyContent: 'center',
  },
  skipButton: {
    paddingHorizontal: Spacing.three,
    paddingVertical: Spacing.one,
    borderRadius: 14,
    backgroundColor: 'rgba(20, 26, 54, 0.6)',
    borderWidth: 1,
    borderColor: 'rgba(169, 176, 204, 0.18)',
  },
  skipButtonPressed: {
    opacity: 0.7,
    backgroundColor: 'rgba(22, 139, 255, 0.15)',
  },
  skipText: {
    color: BrandColors.secondaryText,
    fontSize: 13,
    fontWeight: '600',
    letterSpacing: 0.5,
  },
  skipPlaceholder: {
    width: 52,
  },
});

export default OnboardingHeader;

