import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import Animated, { Easing, FadeInDown } from 'react-native-reanimated';
import Svg, {
  Circle,
  Defs,
  LinearGradient as SvgLinearGradient,
  Path,
  Rect,
  Stop,
} from 'react-native-svg';

import { BrandColors, Spacing } from '@/constants/theme';

interface RecentAnalysesEmptyStateProps {
  onViewAllPress?: () => void;
}

export const RecentAnalysesEmptyState: React.FC<RecentAnalysesEmptyStateProps> = ({
  onViewAllPress,
}) => {
  return (
    <Animated.View
      entering={FadeInDown.delay(350).duration(450).easing(Easing.out(Easing.cubic))}
      style={styles.container}>
      {/* Section Header */}
      <View style={styles.sectionHeader}>
        <Text style={styles.sectionTitle}>Recent Analyses</Text>
        <Pressable
          onPress={onViewAllPress}
          hitSlop={{ top: 8, bottom: 8, left: 10, right: 10 }}
          style={({ pressed }) => [
            styles.viewAllPressable,
            pressed && styles.viewAllPressed,
          ]}>
          <Text style={styles.viewAllText}>View All →</Text>
        </Pressable>
      </View>

      {/* Empty State Card */}
      <View style={styles.emptyCard}>
        {/* Glowing Audio File Icon */}
        <View style={styles.iconWrapper}>
          <Svg width={42} height={42} viewBox="0 0 42 42">
            <Defs>
              <SvgLinearGradient id="audioFileGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <Stop offset="0%" stopColor={BrandColors.electricBlue} />
                <Stop offset="100%" stopColor={BrandColors.primaryPurple} />
              </SvgLinearGradient>
            </Defs>

            {/* Document Contour */}
            <Path
              d="M 12 6 L 24 6 L 32 14 L 32 36 C 32 37.1, 31.1 38, 30 38 L 12 38 C 10.9 38, 10 37.1, 10 36 L 10 8 C 10 6.9, 10.9 6, 12 6 Z"
              fill="#0D1230"
              stroke="url(#audioFileGrad)"
              strokeWidth="1.75"
            />
            {/* Folded Corner */}
            <Path
              d="M 24 6 L 24 14 L 32 14"
              fill="none"
              stroke="url(#audioFileGrad)"
              strokeWidth="1.5"
            />

            {/* Audio Wave Sound Bars on Document */}
            <Rect x="15" y="24" width="2" height="6" rx="1" fill="#00D9FF" />
            <Rect x="19" y="21" width="2" height="12" rx="1" fill="#168BFF" />
            <Rect x="23" y="19" width="2" height="16" rx="1" fill="#C040FF" />
            <Rect x="27" y="23" width="2" height="8" rx="1" fill="#A020F0" />
          </Svg>
        </View>

        {/* Empty State Typography */}
        <Text style={styles.emptyTitle}>No analyses yet</Text>
        <Text style={styles.emptySubtitle}>
          Your analyzed audio results will appear here.
        </Text>
      </View>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '100%',
    paddingHorizontal: Spacing.four,
    marginBottom: Spacing.six,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: Spacing.three,
  },
  sectionTitle: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '800',
    letterSpacing: 0.3,
  },
  viewAllPressable: {
    paddingVertical: 4,
    paddingHorizontal: 8,
  },
  viewAllPressed: {
    opacity: 0.7,
  },
  viewAllText: {
    color: BrandColors.cyanAccent,
    fontSize: 13,
    fontWeight: '700',
    letterSpacing: 0.4,
  },
  emptyCard: {
    backgroundColor: '#141A36',
    borderRadius: 20,
    borderWidth: 1.25,
    borderColor: 'rgba(22, 139, 255, 0.18)',
    paddingVertical: Spacing.five,
    paddingHorizontal: Spacing.four,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.25,
    shadowRadius: 10,
    elevation: 4,
  },
  iconWrapper: {
    width: 64,
    height: 64,
    borderRadius: 20,
    backgroundColor: 'rgba(13, 18, 48, 0.75)',
    borderWidth: 1,
    borderColor: 'rgba(0, 217, 255, 0.25)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.three,
  },
  emptyTitle: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
    letterSpacing: 0.2,
    marginBottom: 4,
  },
  emptySubtitle: {
    color: BrandColors.secondaryText,
    fontSize: 13,
    textAlign: 'center',
    lineHeight: 18,
    maxWidth: 240,
  },
});

export default RecentAnalysesEmptyState;

