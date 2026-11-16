import { useEffect, useState } from 'react';

// In-memory module-level reactive store for onboarding completion state
let hasCompletedOnboardingGlobal = false;
const listeners = new Set<(completed: boolean) => void>();

export function getOnboardingCompleted(): boolean {
  return hasCompletedOnboardingGlobal;
}

export function setOnboardingCompleted(completed: boolean): void {
  hasCompletedOnboardingGlobal = completed;
  listeners.forEach((listener) => listener(completed));
}

export function useOnboardingCompleted(): [boolean, (completed: boolean) => void] {
  const [completed, setCompleted] = useState(hasCompletedOnboardingGlobal);

  useEffect(() => {
    const listener = (newVal: boolean) => setCompleted(newVal);
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  }, []);

  return [completed, setOnboardingCompleted];
}

