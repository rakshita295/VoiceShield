import { DarkTheme, ThemeProvider, Slot } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import React from 'react';
import { GestureHandlerRootView } from 'react-native-gesture-handler';

import { VoiceShieldSplashScreen } from '@/components/branding/VoiceShieldSplashScreen';
import { BrandColors } from '@/constants/theme';

SplashScreen.preventAutoHideAsync();

const VoiceShieldTheme = {
  ...DarkTheme,
  colors: {
    ...DarkTheme.colors,
    background: BrandColors.primaryBackground,
    card: BrandColors.secondaryBackground,
    text: BrandColors.primaryText,
    border: BrandColors.borderMuted,
    primary: BrandColors.electricBlue,
  },
};

export default function RootLayout() {
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <ThemeProvider value={VoiceShieldTheme}>
        <VoiceShieldSplashScreen displayDuration={3000} fadeDuration={500} />
        <Slot />
      </ThemeProvider>
    </GestureHandlerRootView>
  );
}
