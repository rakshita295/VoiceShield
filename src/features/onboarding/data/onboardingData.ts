export interface OnboardingSlide {
  id: string;
  step: number;
  title: string;
  description: string;
  buttonText: string;
  tag: string;
}

export const ONBOARDING_SLIDES: OnboardingSlide[] = [
  {
    id: 'protect-voice',
    step: 1,
    title: 'Protect Your Voice',
    description:
      'Verify the authenticity of any audio and stay safe from voice manipulation.',
    buttonText: 'Next',
    tag: 'VOICE SHIELD ENGINE',
  },
  {
    id: 'detect-ai',
    step: 2,
    title: 'Detect AI Voices',
    description:
      'Identify synthetic and deepfake voices with advanced AI models.',
    buttonText: 'Next',
    tag: 'SPECTRAL AUDIO ANALYSIS',
  },
  {
    id: 'privacy-matters',
    step: 3,
    title: 'Your Privacy Matters',
    description:
      'Analyze audio securely on your device, with no unnecessary data sharing.',
    buttonText: 'Get Started',
    tag: 'SECURE ARCHITECTURE',
  },
];

