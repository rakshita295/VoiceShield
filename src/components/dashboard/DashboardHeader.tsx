import React from 'react';
import { Pressable, StyleSheet, View } from 'react-native';
import Animated, { FadeIn } from 'react-native-reanimated';
import Svg, {
  Circle,
  Defs,
  LinearGradient as SvgLinearGradient,
  Path,
  Stop,
  Text as SvgText,
  TSpan,
} from 'react-native-svg';

import { VoiceShieldLogo } from '@/components/branding/VoiceShieldLogo';
import { BrandColors, Spacing } from '@/constants/theme';

interface DashboardHeaderProps {
  onProfilePress?: () => void;
}

export const DashboardHeader: React.FC<DashboardHeaderProps> = ({
  onProfilePress,
}) => {
  return (
    <Animated.View
      entering={FadeIn.duration(400)}
      style={styles.headerContainer}>
      {/* Left: Logo + Wordmark */}
      <View style={styles.brandRow}>
        <VoiceShieldLogo size={36} animated={false} />
        <Svg height={28} width={138} viewBox="0 0 138 28">
          <Defs>
            <SvgLinearGradient id="dashWordmarkGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <Stop offset="0%" stopColor={BrandColors.electricBlue} />
              <Stop offset="100%" stopColor={BrandColors.cyanAccent} />
            </SvgLinearGradient>
          </Defs>
          <SvgText
            x="0"
            y="22"
            fontSize="22"
            fontWeight="800"
            letterSpacing="0.5">
            <TSpan fill="#FFFFFF">Voice</TSpan>
            <TSpan fill="url(#dashWordmarkGrad)">Shield</TSpan>
          </SvgText>
        </Svg>
      </View>

      {/* Right: Circular Profile Icon with Subtle Purple-Blue Border */}
      <Pressable
        onPress={onProfilePress}
        hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
        style={({ pressed }) => [
          styles.profileButton,
          pressed && styles.profileButtonPressed,
        ]}>
        <Svg width={38} height={38} viewBox="0 0 38 38">
          <Defs>
            <SvgLinearGradient id="profileBorderGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <Stop offset="0%" stopColor={BrandColors.primaryPurple} />
              <Stop offset="100%" stopColor={BrandColors.electricBlue} />
            </SvgLinearGradient>
          </Defs>
          {/* Subtle Cyber Gradient Border */}
          <Circle
            cx="19"
            cy="19"
            r="17.5"
            stroke="url(#profileBorderGrad)"
            strokeWidth="1.5"
            fill="#141A36"
          />
          {/* User Silhouette Icon */}
          <Circle cx="19" cy="14" r="5" fill="#A9B0CC" />
          <Path
            d="M 10.5 28 C 10.5 23.5, 14 21, 19 21 C 24 21, 27.5 23.5, 27.5 28"
            stroke="#A9B0CC"
            strokeWidth="2"
            strokeLinecap="round"
            fill="none"
          />
        </Svg>
      </Pressable>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  headerContainer: {
    width: '100%',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: Spacing.four,
    paddingTop: Spacing.two,
    paddingBottom: Spacing.three,
  },
  brandRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.two,
  },
  profileButton: {
    width: 38,
    height: 38,
    borderRadius: 19,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: BrandColors.electricBlue,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 6,
    elevation: 3,
  },
  profileButtonPressed: {
    opacity: 0.75,
    transform: [{ scale: 0.95 }],
  },
});

export default DashboardHeader;

