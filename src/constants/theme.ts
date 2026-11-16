/**
 * VoiceShield Design System & Brand Identity Tokens
 * Premium Dark Cyber-Security Theme
 */

import '@/global.css';

import { Platform } from 'react-native';

export const BrandColors = {
  primaryBackground: '#070817',
  secondaryBackground: '#0D1230',
  cardBackground: '#141A36',
  primaryPurple: '#A020F0',
  electricBlue: '#168BFF',
  cyanAccent: '#00D9FF',
  primaryText: '#FFFFFF',
  secondaryText: '#A9B0CC',
  glowPurple: 'rgba(160, 32, 240, 0.45)',
  glowBlue: 'rgba(22, 139, 255, 0.45)',
  glowCyan: 'rgba(0, 217, 255, 0.45)',
  borderMuted: 'rgba(169, 176, 204, 0.15)',
} as const;

export const BrandGradients = {
  background: ['#070817', '#0D1230'] as const,
  backgroundFull: ['#04050E', '#070817', '#0D1230'] as const,
  shieldBorder: ['#A020F0', '#168BFF'] as const,
  shieldBorderFull: ['#A020F0', '#168BFF', '#00D9FF'] as const,
  electricCyan: ['#168BFF', '#00D9FF'] as const,
  waveform: ['#A020F0', '#168BFF', '#00D9FF'] as const,
  card: ['#141A36', '#0D1230'] as const,
} as const;

export const Colors = {
  light: {
    text: '#070817',
    background: '#F4F6FC',
    backgroundElement: '#E7ECF7',
    backgroundSelected: '#D9E2F4',
    textSecondary: '#5A627A',
    primaryPurple: BrandColors.primaryPurple,
    electricBlue: BrandColors.electricBlue,
    cyanAccent: BrandColors.cyanAccent,
  },
  dark: {
    text: BrandColors.primaryText,
    background: BrandColors.primaryBackground,
    backgroundElement: BrandColors.secondaryBackground,
    backgroundSelected: BrandColors.cardBackground,
    textSecondary: BrandColors.secondaryText,
    primaryPurple: BrandColors.primaryPurple,
    electricBlue: BrandColors.electricBlue,
    cyanAccent: BrandColors.cyanAccent,
  },
} as const;

export type ThemeColor = keyof typeof Colors.light & keyof typeof Colors.dark;

export const Fonts = Platform.select({
  ios: {
    sans: 'system-ui',
    serif: 'ui-serif',
    rounded: 'ui-rounded',
    mono: 'ui-monospace',
  },
  default: {
    sans: 'normal',
    serif: 'serif',
    rounded: 'normal',
    mono: 'monospace',
  },
  web: {
    sans: 'var(--font-display)',
    serif: 'var(--font-serif)',
    rounded: 'var(--font-rounded)',
    mono: 'var(--font-mono)',
  },
});

export const Spacing = {
  half: 2,
  one: 4,
  two: 8,
  three: 16,
  four: 24,
  five: 32,
  six: 64,
} as const;

export const BottomTabInset = Platform.select({ ios: 50, android: 80 }) ?? 0;
export const MaxContentWidth = 800;
