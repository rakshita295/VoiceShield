import { router } from 'expo-router';
import { useCallback, useEffect, useState } from 'react';
import { BackHandler } from 'react-native';

import { ONBOARDING_SLIDES } from '@/features/onboarding/data/onboardingData';
import { setOnboardingCompleted } from '@/features/onboarding/data/onboardingStore';

export function useOnboarding(onComplete?: () => void) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isTransitioning, setIsTransitioning] = useState(false);

  const totalSlides = ONBOARDING_SLIDES.length;
  const isFirstSlide = currentIndex === 0;
  const isLastSlide = currentIndex === totalSlides - 1;
  const currentSlide = ONBOARDING_SLIDES[currentIndex];

  const completeOnboarding = useCallback(() => {
    setOnboardingCompleted(true);
    onComplete?.();
    // Navigate to the existing home route
    router.replace('/');
  }, [onComplete]);

  const goToNext = useCallback(() => {
    if (isTransitioning) return;

    if (isLastSlide) {
      completeOnboarding();
      return;
    }

    setIsTransitioning(true);
    setCurrentIndex((prev) => Math.min(prev + 1, totalSlides - 1));
    setTimeout(() => setIsTransitioning(false), 450);
  }, [isTransitioning, isLastSlide, totalSlides, completeOnboarding]);

  const goToPrev = useCallback(() => {
    if (isTransitioning || isFirstSlide) return false;

    setIsTransitioning(true);
    setCurrentIndex((prev) => Math.max(prev - 1, 0));
    setTimeout(() => setIsTransitioning(false), 450);
    return true;
  }, [isTransitioning, isFirstSlide]);

  const skip = useCallback(() => {
    if (isTransitioning) return;
    completeOnboarding();
  }, [isTransitioning, completeOnboarding]);

  // Handle physical Android hardware back button
  useEffect(() => {
    const onBackPress = () => {
      if (currentIndex > 0) {
        goToPrev();
        return true; // Intercept Android back press to go to previous slide
      }
      return false; // Let OS handle back on first slide
    };

    const backHandlerSubscription = BackHandler.addEventListener(
      'hardwareBackPress',
      onBackPress
    );

    return () => backHandlerSubscription.remove();
  }, [currentIndex, goToPrev]);

  return {
    currentIndex,
    currentSlide,
    totalSlides,
    isFirstSlide,
    isLastSlide,
    isTransitioning,
    goToNext,
    goToPrev,
    skip,
    completeOnboarding,
  };
}

export default useOnboarding;
