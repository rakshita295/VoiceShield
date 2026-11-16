import { router } from 'expo-router';
import React from 'react';

import { useOnboardingCompleted } from '@/features/onboarding/data/onboardingStore';
import { OnboardingScreen } from '@/features/onboarding/components/OnboardingScreen';

export default function OnboardingRoute() {
  const [, setHasCompletedOnboarding] = useOnboardingCompleted();

  const handleComplete = () => {
    setHasCompletedOnboarding(true);
    router.replace('/');
  };

  return <OnboardingScreen onComplete={handleComplete} />;
}
