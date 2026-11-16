import { LinearGradient } from 'expo-linear-gradient';
import { StatusBar } from 'expo-status-bar';
import React, { useEffect, useState } from 'react';
import { BackHandler, StyleSheet, View } from 'react-native';
import Svg, { Circle, Defs, RadialGradient, Stop } from 'react-native-svg';

import { TabKey, VoiceShieldBottomNav } from '@/components/navigation/VoiceShieldBottomNav';
import { HistoryPlaceholderView } from '@/features/dashboard/components/HistoryPlaceholderView';
import { HomeDashboardView } from '@/features/dashboard/components/HomeDashboardView';
import { InsightsPlaceholderView } from '@/features/dashboard/components/InsightsPlaceholderView';
import { SettingsPlaceholderView } from '@/features/dashboard/components/SettingsPlaceholderView';
import { BrandColors } from '@/constants/theme';

interface VoiceShieldMainShellProps {
  initialTab?: TabKey;
}

export const VoiceShieldMainShell: React.FC<VoiceShieldMainShellProps> = ({
  initialTab = 'home',
}) => {
  const [activeTab, setActiveTab] = useState<TabKey>(initialTab);

  // Handle Android hardware back button: return to 'home' tab if on other tabs
  useEffect(() => {
    const onBackPress = () => {
      if (activeTab !== 'home') {
        setActiveTab('home');
        return true; // Intercept and return to home
      }
      return false; // Exit app or default system back
    };

    const backHandlerSubscription = BackHandler.addEventListener(
      'hardwareBackPress',
      onBackPress
    );

    return () => backHandlerSubscription.remove();
  }, [activeTab]);

  const renderActiveTabContent = () => {
    switch (activeTab) {
      case 'home':
        return <HomeDashboardView onNavigateToTab={(tab) => setActiveTab(tab)} />;
      case 'history':
        return <HistoryPlaceholderView onGoHome={() => setActiveTab('home')} />;
      case 'insights':
        return <InsightsPlaceholderView onGoHome={() => setActiveTab('home')} />;
      case 'settings':
        return <SettingsPlaceholderView onGoHome={() => setActiveTab('home')} />;
      default:
        return <HomeDashboardView onNavigateToTab={(tab) => setActiveTab(tab)} />;
    }
  };

  return (
    <View style={styles.container}>
      <StatusBar style="light" />

      {/* Deep Navy/Black Background Gradient */}
      <LinearGradient
        colors={[BrandColors.primaryBackground, BrandColors.secondaryBackground, '#050713']}
        start={{ x: 0.5, y: 0 }}
        end={{ x: 0.5, y: 1 }}
        style={StyleSheet.absoluteFill}
      />

      {/* Ambient Radial Background Glow */}
      <View pointerEvents="none" style={styles.ambientGlowWrapper}>
        <Svg width="100%" height="100%" viewBox="0 0 360 360">
          <Defs>
            <RadialGradient id="shellGlow" cx="50%" cy="40%" r="50%">
              <Stop offset="0%" stopColor={BrandColors.electricBlue} stopOpacity="0.22" />
              <Stop offset="45%" stopColor={BrandColors.primaryPurple} stopOpacity="0.14" />
              <Stop offset="80%" stopColor={BrandColors.cyanAccent} stopOpacity="0.03" />
              <Stop offset="100%" stopColor="#070817" stopOpacity="0" />
            </RadialGradient>
          </Defs>
          <Circle cx="180" cy="180" r="180" fill="url(#shellGlow)" />
        </Svg>
      </View>

      {/* Active Tab Screen Content */}
      <View style={styles.contentLayer}>{renderActiveTabContent()}</View>

      {/* Reusable Fixed 4-Tab Bottom Navigation Bar */}
      <VoiceShieldBottomNav
        activeTab={activeTab}
        onTabChange={(tab) => setActiveTab(tab)}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: BrandColors.primaryBackground,
  },
  ambientGlowWrapper: {
    position: 'absolute',
    top: '10%',
    left: 0,
    right: 0,
    height: 360,
    alignItems: 'center',
    justifyContent: 'center',
  },
  contentLayer: {
    flex: 1,
  },
});

export default VoiceShieldMainShell;

