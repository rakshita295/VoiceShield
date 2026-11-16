import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import Animated, { Easing, FadeInDown } from 'react-native-reanimated';

import { BrandColors, Spacing } from '@/constants/theme';

export const WelcomeSection: React.FC = () => {
  return (
    <Animated.View
      entering={FadeInDown.delay(100).duration(450).easing(Easing.out(Easing.cubic))}
      style={styles.container}>
      <Text style={styles.headingText}>Welcome Back!</Text>
      <Text style={styles.descriptionText}>
        Verify, analyze, and protect your voice.
      </Text>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '100%',
    paddingHorizontal: Spacing.four,
    paddingTop: Spacing.two,
    paddingBottom: Spacing.four,
    gap: 4,
  },
  headingText: {
    color: '#FFFFFF',
    fontSize: 28,
    fontWeight: '800',
    letterSpacing: 0.3,
  },
  descriptionText: {
    color: BrandColors.secondaryText,
    fontSize: 14.5,
    fontWeight: '500',
    lineHeight: 20,
    letterSpacing: 0.2,
  },
});

export default WelcomeSection;

