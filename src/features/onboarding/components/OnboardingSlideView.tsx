import React from 'react';
import { Dimensions, StyleSheet, Text, View } from 'react-native';
import Animated, {
  Easing,
  FadeInDown,
  FadeInUp,
} from 'react-native-reanimated';

import { DetectAiIllustration } from '@/features/onboarding/components/DetectAiIllustration';
import { PrivacyMattersIllustration } from '@/features/onboarding/components/PrivacyMattersIllustration';
import { ProtectVoiceIllustration } from '@/features/onboarding/components/ProtectVoiceIllustration';
import { OnboardingSlide } from '@/features/onboarding/data/onboardingData';
import { BrandColors, Spacing } from '@/constants/theme';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

interface OnboardingSlideViewProps {
  slide: OnboardingSlide;
}

export const OnboardingSlideView: React.FC<OnboardingSlideViewProps> = ({ slide }) => {
  // Render the matching code-generated vector illustration based on slide step
  const renderIllustration = () => {
    const illustrationSize = Math.min(SCREEN_WIDTH * 0.82, 320);

    switch (slide.step) {
      case 1:
        return <ProtectVoiceIllustration size={illustrationSize} />;
      case 2:
        return <DetectAiIllustration size={illustrationSize} />;
      case 3:
        return <PrivacyMattersIllustration size={illustrationSize} />;
      default:
        return null;
    }
  };

  return (
    <View style={styles.slideContainer}>
      {/* Upper Area: Center Vector Illustration */}
      <View style={styles.illustrationArea}>
        <Animated.View
          key={`illus-${slide.id}`}
          entering={FadeInUp.duration(600).easing(Easing.out(Easing.cubic))}
          style={styles.illustrationWrapper}>
          {renderIllustration()}
        </Animated.View>
      </View>

      {/* Lower Area: Content Block (Tag + Heading + Description) */}
      <Animated.View
        key={`content-${slide.id}`}
        entering={FadeInDown.delay(180).duration(550).easing(Easing.out(Easing.cubic))}
        style={styles.textContainer}>
        {/* Subtle Cyber Category Tag */}
        <View style={styles.tagBadge}>
          <View style={styles.tagDot} />
          <Text style={styles.tagText}>{slide.tag}</Text>
        </View>

        {/* Slide Heading */}
        <Text style={styles.titleText}>{slide.title}</Text>

        {/* Slide Description */}
        <Text style={styles.descriptionText}>{slide.description}</Text>
      </Animated.View>
    </View>
  );
};

const styles = StyleSheet.create({
  slideContainer: {
    width: SCREEN_WIDTH,
    flex: 1,
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: Spacing.four,
  },
  illustrationArea: {
    flex: 1.3,
    width: '100%',
    alignItems: 'center',
    justifyContent: 'center',
  },
  illustrationWrapper: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  textContainer: {
    flex: 0.9,
    width: '100%',
    alignItems: 'center',
    justifyContent: 'flex-start',
    paddingHorizontal: Spacing.three,
    gap: Spacing.two,
  },
  tagBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 14,
    backgroundColor: 'rgba(22, 139, 255, 0.08)',
    borderWidth: 1,
    borderColor: 'rgba(0, 217, 255, 0.22)',
    marginBottom: Spacing.one,
  },
  tagDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: BrandColors.cyanAccent,
  },
  tagText: {
    color: BrandColors.cyanAccent,
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 1.2,
  },
  titleText: {
    color: '#FFFFFF',
    fontSize: 30,
    fontWeight: '800',
    letterSpacing: 0.5,
    textAlign: 'center',
  },
  descriptionText: {
    color: BrandColors.secondaryText,
    fontSize: 15,
    lineHeight: 23,
    textAlign: 'center',
    maxWidth: 320,
    marginTop: Spacing.one,
  },
});

export default OnboardingSlideView;

