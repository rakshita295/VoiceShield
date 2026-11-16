import { LinearGradient } from 'expo-linear-gradient';
import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import Svg, { Path } from 'react-native-svg';

import { BrandColors, Spacing } from '@/constants/theme';

interface OnboardingFooterProps {
  currentIndex: number;
  totalSlides: number;
  isLastSlide: boolean;
  isTransitioning: boolean;
  onNext: () => void;
  onPrev: () => void;
}

export const OnboardingFooter: React.FC<OnboardingFooterProps> = ({
  currentIndex,
  totalSlides,
  isLastSlide,
  isTransitioning,
  onNext,
  onPrev,
}) => {
  return (
    <View style={styles.footerContainer}>
      {/* Top Row: Progress Indicator Dots */}
      <View style={styles.dotsRow}>
        {Array.from({ length: totalSlides }).map((_, index) => {
          const isActive = index === currentIndex;
          return (
            <View
              key={index}
              style={[
                styles.dotBase,
                isActive ? styles.dotActive : styles.dotInactive,
              ]}>
              {isActive && (
                <LinearGradient
                  colors={[BrandColors.primaryPurple, BrandColors.electricBlue, BrandColors.cyanAccent]}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={StyleSheet.absoluteFill}
                />
              )}
            </View>
          );
        })}
      </View>

      {/* Bottom Row: Actions (Back + Gradient Next/Get Started Button) */}
      <View style={styles.actionRow}>
        {/* Back Button (Only visible after slide 1) */}
        {currentIndex > 0 ? (
          <Pressable
            onPress={onPrev}
            disabled={isTransitioning}
            hitSlop={{ top: 10, bottom: 10, left: 14, right: 14 }}
            style={({ pressed }) => [
              styles.backButton,
              pressed && styles.backButtonPressed,
            ]}>
            <Svg width={20} height={20} viewBox="0 0 24 24" fill="none">
              <Path
                d="M 15 18 L 9 12 L 15 6"
                stroke={BrandColors.secondaryText}
                strokeWidth={2.5}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </Svg>
            <Text style={styles.backButtonText}>Back</Text>
          </Pressable>
        ) : (
          <View style={styles.backButtonPlaceholder} />
        )}

        {/* Primary Action Button (Next / Get Started) */}
        <Pressable
          onPress={onNext}
          disabled={isTransitioning}
          style={({ pressed }) => [
            styles.primaryButtonWrapper,
            pressed && styles.primaryButtonPressed,
          ]}>
          <LinearGradient
            colors={
              isLastSlide
                ? [BrandColors.primaryPurple, BrandColors.electricBlue, BrandColors.cyanAccent]
                : [BrandColors.primaryPurple, BrandColors.electricBlue]
            }
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={styles.primaryButtonGradient}>
            <Text style={styles.primaryButtonText}>
              {isLastSlide ? 'Get Started' : 'Next'}
            </Text>

            {/* Forward Arrow Icon */}
            <Svg width={18} height={18} viewBox="0 0 24 24" fill="none">
              <Path
                d="M 9 6 L 15 12 L 9 18"
                stroke="#FFFFFF"
                strokeWidth={2.5}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </Svg>
          </LinearGradient>
        </Pressable>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  footerContainer: {
    width: '100%',
    paddingHorizontal: Spacing.four,
    paddingBottom: Spacing.three,
    gap: Spacing.three,
  },
  dotsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.two,
  },
  dotBase: {
    height: 8,
    borderRadius: 4,
    overflow: 'hidden',
  },
  dotActive: {
    width: 28,
  },
  dotInactive: {
    width: 8,
    backgroundColor: '#1C2346',
  },
  actionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    width: '100%',
  },
  backButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: Spacing.two,
    paddingHorizontal: Spacing.three,
    borderRadius: 24,
    gap: 4,
    backgroundColor: 'rgba(20, 26, 54, 0.5)',
    borderWidth: 1,
    borderColor: 'rgba(169, 176, 204, 0.15)',
  },
  backButtonPressed: {
    opacity: 0.7,
    backgroundColor: 'rgba(22, 139, 255, 0.15)',
  },
  backButtonText: {
    color: BrandColors.secondaryText,
    fontSize: 14,
    fontWeight: '600',
  },
  backButtonPlaceholder: {
    width: 78,
  },
  primaryButtonWrapper: {
    borderRadius: 28,
    overflow: 'hidden',
    shadowColor: BrandColors.electricBlue,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.35,
    shadowRadius: 12,
    elevation: 6,
  },
  primaryButtonPressed: {
    opacity: 0.85,
    transform: [{ scale: 0.98 }],
  },
  primaryButtonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    paddingHorizontal: 26,
    borderRadius: 28,
    gap: Spacing.two,
    minWidth: 130,
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
});

export default OnboardingFooter;

