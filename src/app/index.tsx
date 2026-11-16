import React from 'react';

import { useOnboardingCompleted } from '@/features/onboarding/data/onboardingStore';
import { OnboardingScreen } from '@/features/onboarding/components/OnboardingScreen';
import { VoiceShieldMainShell } from '@/features/dashboard/components/VoiceShieldMainShell';

export default function HomeScreen() {
  const [hasCompletedOnboarding, setHasCompletedOnboarding] = useOnboardingCompleted();

  if (!hasCompletedOnboarding) {
    return (
      <OnboardingScreen
        onComplete={() => {
          setHasCompletedOnboarding(true);
        }}
      />
    );
  }

  return <VoiceShieldMainShell initialTab="home" />;
}
