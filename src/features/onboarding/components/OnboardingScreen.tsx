import { LinearGradient } from 'expo-linear-gradient';
import { StatusBar } from 'expo-status-bar';
import React, { useEffect, useMemo } from 'react';
import { Dimensions, StyleSheet, View } from 'react-native';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import Animated, {
  Easing,
  runOnJS,
  useAnimatedStyle,
  useSharedValue,
  withTiming,
} from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Svg, { Circle, Defs, RadialGradient, Stop } from 'react-native-svg';

import { OnboardingFooter } from '@/components/onboarding/OnboardingFooter';
import { OnboardingHeader } from '@/components/onboarding/OnboardingHeader';
import { OnboardingSlideView } from '@/features/onboarding/components/OnboardingSlideView';
import { ONBOARDING_SLIDES } from '@/features/onboarding/data/onboardingData';
import { useOnboarding } from '@/features/onboarding/hooks/useOnboarding';
import { BrandColors } from '@/constants/theme';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

// Thresholds to distinguish intentional swipes from accidental touches
const SWIPE_THRESHOLD = 50;
const VELOCITY_THRESHOLD = 400;

export interface OnboardingScreenProps {
  onComplete?: () => void;
}

export const OnboardingScreen: React.FC<OnboardingScreenProps> = ({ onComplete }) => {
  const insets = useSafeAreaInsets();
  const {
    currentIndex,
    totalSlides,
    isLastSlide,
    isTransitioning,
    goToNext,
    goToPrev,
    skip,
  } = useOnboarding(onComplete);

  // Reanimated shared values
  const translateX = useSharedValue(0);
  const dragX = useSharedValue(0);
  const currentIndexShared = useSharedValue(currentIndex);
  const isTransitioningShared = useSharedValue(isTransitioning);

  // Synchronize shared values with React state for worklet access
  useEffect(() => {
    currentIndexShared.value = currentIndex;
  }, [currentIndex, currentIndexShared]);

  useEffect(() => {
    isTransitioningShared.value = isTransitioning;
  }, [isTransitioning, isTransitioningShared]);

  // Smooth slide translation when currentIndex updates
  useEffect(() => {
    dragX.value = 0;
    translateX.value = withTiming(-currentIndex * SCREEN_WIDTH, {
      duration: 380,
      easing: Easing.out(Easing.cubic),
    });
  }, [currentIndex, translateX, dragX]);

  // Combined animated style: settled slide offset + real-time touch displacement
  const animatedCarouselStyle = useAnimatedStyle(() => ({
    transform: [{ translateX: translateX.value + dragX.value }],
  }));

  // Pan gesture recognizer for natural swipe navigation
  const panGesture = useMemo(() => {
    return Gesture.Pan()
      // Only activate when horizontal movement exceeds 20px (prevents accidental taps)
      .activeOffsetX([-20, 20])
      // Fail if vertical movement dominates (> 25px) to prevent vertical conflicts
      .failOffsetY([-25, 25])
      .onUpdate((event) => {
        'worklet';
        if (isTransitioningShared.value) return;

        let translation = event.translationX;

        // Boundary restriction: prevent swiping right on the first screen
        if (currentIndexShared.value === 0 && translation > 0) {
          translation = translation * 0.15; // rubber-band resistance
        }
        // Boundary restriction: prevent swiping left on the last screen
        else if (currentIndexShared.value === totalSlides - 1 && translation < 0) {
          translation = translation * 0.15; // rubber-band resistance
        }

        dragX.value = translation;
      })
      .onEnd((event) => {
        'worklet';
        if (isTransitioningShared.value) return;

        const isSwipeLeft =
          event.translationX < -SWIPE_THRESHOLD ||
          event.velocityX < -VELOCITY_THRESHOLD;
        const isSwipeRight =
          event.translationX > SWIPE_THRESHOLD ||
          event.velocityX > VELOCITY_THRESHOLD;

        // 1. Swiping left -> Navigate to next screen
        if (isSwipeLeft && currentIndexShared.value < totalSlides - 1) {
          dragX.value = withTiming(0, { duration: 180, easing: Easing.out(Easing.cubic) });
          runOnJS(goToNext)();
        }
        // 2. Swiping right -> Navigate to previous screen
        else if (isSwipeRight && currentIndexShared.value > 0) {
          dragX.value = withTiming(0, { duration: 180, easing: Easing.out(Easing.cubic) });
          runOnJS(goToPrev)();
        }
        // 3. Sub-threshold drag or boundary hit -> Rebound to current screen
        else {
          dragX.value = withTiming(0, {
            duration: 250,
            easing: Easing.out(Easing.cubic),
          });
        }
      })
      .onFinalize(() => {
        'worklet';
        dragX.value = withTiming(0, {
          duration: 250,
          easing: Easing.out(Easing.cubic),
        });
      });
  }, [currentIndexShared, isTransitioningShared, totalSlides, goToNext, goToPrev, dragX]);

  return (
    <View style={styles.rootContainer}>
      <StatusBar style="light" />

      {/* Full-screen Deep Navy / Black Gradient Background */}
      <LinearGradient
        colors={[BrandColors.primaryBackground, BrandColors.secondaryBackground, '#050713']}
        start={{ x: 0.5, y: 0 }}
        end={{ x: 0.5, y: 1 }}
        style={StyleSheet.absoluteFill}
      />

      {/* Background Ambient Radial Glow */}
      <View pointerEvents="none" style={styles.bgGlowWrapper}>
        <Svg width={SCREEN_WIDTH} height={SCREEN_WIDTH} viewBox="0 0 360 360">
          <Defs>
            <RadialGradient id="onboardingGlow" cx="50%" cy="50%" r="50%">
              <Stop offset="0%" stopColor={BrandColors.electricBlue} stopOpacity="0.28" />
              <Stop offset="40%" stopColor={BrandColors.primaryPurple} stopOpacity="0.18" />
              <Stop offset="75%" stopColor={BrandColors.cyanAccent} stopOpacity="0.04" />
              <Stop offset="100%" stopColor="#070817" stopOpacity="0" />
            </RadialGradient>
          </Defs>
          <Circle cx="180" cy="180" r="180" fill="url(#onboardingGlow)" />
        </Svg>
      </View>

      {/* Main Safe Area Layout */}
      <View
        style={[
          styles.safeAreaContainer,
          {
            paddingTop: Math.max(insets.top, 32),
            paddingBottom: Math.max(insets.bottom, 20),
          },
        ]}>
        {/* Top Header: Logo + Skip Button */}
        <OnboardingHeader onSkip={skip} showSkip={!isLastSlide} />

        {/* Center: Horizontal Sliding Carousel with Swipe Gesture Detector */}
        <GestureDetector gesture={panGesture}>
          <View style={styles.carouselContainer}>
            <Animated.View style={[styles.slidesRow, animatedCarouselStyle]}>
              {ONBOARDING_SLIDES.map((slide) => (
                <OnboardingSlideView key={slide.id} slide={slide} />
              ))}
            </Animated.View>
          </View>
        </GestureDetector>

        {/* Bottom Footer: Progress Dots + Actions (Isolated from Pan Gesture) */}
        <OnboardingFooter
          currentIndex={currentIndex}
          totalSlides={totalSlides}
          isLastSlide={isLastSlide}
          isTransitioning={isTransitioning}
          onNext={goToNext}
          onPrev={goToPrev}
        />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  rootContainer: {
    flex: 1,
    backgroundColor: BrandColors.primaryBackground,
  },
  bgGlowWrapper: {
    position: 'absolute',
    top: '15%',
    left: 0,
    right: 0,
    alignItems: 'center',
    justifyContent: 'center',
  },
  safeAreaContainer: {
    flex: 1,
    justifyContent: 'space-between',
  },
  carouselContainer: {
    flex: 1,
    width: SCREEN_WIDTH,
    overflow: 'hidden',
    justifyContent: 'center',
  },
  slidesRow: {
    flex: 1,
    flexDirection: 'row',
    width: SCREEN_WIDTH * 3,
  },
});

export default OnboardingScreen;
