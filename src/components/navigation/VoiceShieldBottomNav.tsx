import { LinearGradient } from 'expo-linear-gradient';
import React from 'react';
import { Dimensions, Platform, Pressable, StyleSheet, Text, View } from 'react-native';
import Animated, {
  Easing,
  useAnimatedStyle,
  withSpring,
  withTiming,
} from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Svg, {
  Circle,
  Defs,
  Line,
  LinearGradient as SvgLinearGradient,
  Path,
  Rect,
  Stop,
} from 'react-native-svg';

import { BrandColors, Spacing } from '@/constants/theme';

export type TabKey = 'home' | 'history' | 'insights' | 'settings';

interface TabItem {
  key: TabKey;
  label: string;
}

const TABS: TabItem[] = [
  { key: 'home', label: 'Home' },
  { key: 'history', label: 'History' },
  { key: 'insights', label: 'Insights' },
  { key: 'settings', label: 'Settings' },
];

interface VoiceShieldBottomNavProps {
  activeTab: TabKey;
  onTabChange: (tab: TabKey) => void;
}

export const VoiceShieldBottomNav: React.FC<VoiceShieldBottomNavProps> = ({
  activeTab,
  onTabChange,
}) => {
  const insets = useSafeAreaInsets();
  const bottomInset = Math.max(insets.bottom, 12);

  const renderIcon = (tab: TabKey, isActive: boolean) => {
    const activeColor = BrandColors.cyanAccent;
    const inactiveColor = '#8A92B2';
    const strokeColor = isActive ? activeColor : inactiveColor;
    const fillColor = isActive ? 'rgba(0, 217, 255, 0.15)' : 'none';

    switch (tab) {
      case 'home':
        return (
          <Svg width={24} height={24} viewBox="0 0 24 24" fill="none">
            <Path
              d="M 3 10.5 L 12 3 L 21 10.5 L 21 20 C 21 20.6, 20.6 21, 20 21 L 14 21 L 14 14 L 10 14 L 10 21 L 4 21 C 3.4 21, 3 20.6, 3 20 Z"
              stroke={strokeColor}
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
              fill={fillColor}
            />
          </Svg>
        );

      case 'history':
        return (
          <Svg width={24} height={24} viewBox="0 0 24 24" fill="none">
            {/* Clock Circle with Counter-Clockwise History Arc */}
            <Circle cx="12" cy="12" r="9" stroke={strokeColor} strokeWidth={2} fill={fillColor} />
            <Path
              d="M 12 7 L 12 12 L 15.5 14"
              stroke={strokeColor}
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </Svg>
        );

      case 'insights':
        return (
          <Svg width={24} height={24} viewBox="0 0 24 24" fill="none">
            {/* Analytics / Bar Chart */}
            <Rect x="4" y="14" width="3.5" height="7" rx="1" fill={strokeColor} />
            <Rect x="10.25" y="8" width="3.5" height="13" rx="1" fill={strokeColor} />
            <Rect x="16.5" y="4" width="3.5" height="17" rx="1" fill={strokeColor} />
            {/* Trendline Accent */}
            <Path
              d="M 4 10 L 11 5 L 18 2"
              stroke={isActive ? '#FFFFFF' : 'none'}
              strokeWidth={1.5}
              strokeLinecap="round"
            />
          </Svg>
        );

      case 'settings':
        return (
          <Svg width={24} height={24} viewBox="0 0 24 24" fill="none">
            {/* Cog / Gear */}
            <Circle cx="12" cy="12" r="3.2" stroke={strokeColor} strokeWidth={2} fill={fillColor} />
            <Path
              d="M 19.4 15 A 1.65 1.65 0 0 0 19.7 16.8 L 19.8 16.9 A 2 2 0 1 1 17 19.7 L 16.9 19.6 A 1.65 1.65 0 0 0 15 19.3 A 1.65 1.65 0 0 0 13.9 20.8 L 13.9 21 A 2 2 0 1 1 9.9 21 L 9.9 20.8 A 1.65 1.65 0 0 0 8.8 19.3 A 1.65 1.65 0 0 0 7 19.6 L 6.9 19.7 A 2 2 0 1 1 4.1 16.9 L 4.2 16.8 A 1.65 1.65 0 0 0 4.5 15 A 1.65 1.65 0 0 0 3 13.9 L 2.8 13.9 A 2 2 0 1 1 2.8 9.9 L 3 9.9 A 1.65 1.65 0 0 0 4.5 8.8 A 1.65 1.65 0 0 0 4.2 7 L 4.1 6.9 A 2 2 0 1 1 6.9 4.1 L 7 4.2 A 1.65 1.65 0 0 0 8.8 4.5 A 1.65 1.65 0 0 0 9.9 3 L 9.9 2.8 A 2 2 0 1 1 13.9 2.8 L 13.9 3 A 1.65 1.65 0 0 0 15 4.5 A 1.65 1.65 0 0 0 16.8 4.2 L 16.9 4.1 A 2 2 0 1 1 19.7 6.9 L 19.6 7 A 1.65 1.65 0 0 0 19.3 8.8 A 1.65 1.65 0 0 0 20.8 9.9 L 21 9.9 A 2 2 0 1 1 21 13.9 L 20.8 13.9 A 1.65 1.65 0 0 0 19.4 15 Z"
              stroke={strokeColor}
              strokeWidth={1.8}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </Svg>
        );
    }
  };

  return (
    <View style={[styles.bottomBarContainer, { paddingBottom: bottomInset }]}>
      {/* Translucent Dark Gradient Background with Top Border */}
      <LinearGradient
        colors={['rgba(13, 18, 48, 0.94)', 'rgba(7, 8, 23, 0.98)']}
        start={{ x: 0, y: 0 }}
        end={{ x: 0, y: 1 }}
        style={StyleSheet.absoluteFill}
      />
      {/* Subtle Glowing Cyan/Blue Top Border Line */}
      <View style={styles.topBorderLine} />

      {/* Tabs Row */}
      <View style={styles.tabsRow}>
        {TABS.map((tab) => {
          const isActive = tab.key === activeTab;
          return (
            <Pressable
              key={tab.key}
              onPress={() => onTabChange(tab.key)}
              hitSlop={{ top: 8, bottom: 8, left: 12, right: 12 }}
              style={({ pressed }) => [
                styles.tabItem,
                pressed && styles.tabItemPressed,
              ]}>
              {/* Active Tab Glow Pill Indicator */}
              {isActive && (
                <View style={styles.activeGlowPill}>
                  <LinearGradient
                    colors={['rgba(160, 32, 240, 0.25)', 'rgba(22, 139, 255, 0.25)']}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 0 }}
                    style={StyleSheet.absoluteFill}
                  />
                </View>
              )}

              {/* Icon */}
              <View style={styles.iconContainer}>{renderIcon(tab.key, isActive)}</View>

              {/* Label */}
              <Text style={[styles.tabLabel, isActive ? styles.labelActive : styles.labelInactive]}>
                {tab.label}
              </Text>

              {/* Active Dot Accent */}
              {isActive && <View style={styles.activeDot} />}
            </Pressable>
          );
        })}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  bottomBarContainer: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    borderTopLeftRadius: 22,
    borderTopRightRadius: 22,
    overflow: 'hidden',
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: -6 },
    shadowOpacity: 0.4,
    shadowRadius: 14,
    elevation: 12,
  },
  topBorderLine: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 1.25,
    backgroundColor: 'rgba(22, 139, 255, 0.25)',
  },
  tabsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-around',
    paddingTop: Spacing.two,
    paddingHorizontal: Spacing.two,
  },
  tabItem: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 6,
    position: 'relative',
  },
  tabItemPressed: {
    opacity: 0.75,
  },
  activeGlowPill: {
    position: 'absolute',
    top: 2,
    width: 54,
    height: 38,
    borderRadius: 19,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(0, 217, 255, 0.3)',
  },
  iconContainer: {
    width: 28,
    height: 28,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 2,
  },
  tabLabel: {
    fontSize: 11,
    fontWeight: '600',
    letterSpacing: 0.2,
  },
  labelActive: {
    color: '#FFFFFF',
    fontWeight: '700',
  },
  labelInactive: {
    color: '#8A92B2',
  },
  activeDot: {
    width: 4,
    height: 4,
    borderRadius: 2,
    backgroundColor: BrandColors.cyanAccent,
    marginTop: 2,
  },
});

export default VoiceShieldBottomNav;

