import { router } from 'expo-router';
import React, { useState } from 'react';
import { ScrollView, StyleSheet, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { DashboardHeader } from '@/components/dashboard/DashboardHeader';
import { MainAuthenticityCard } from '@/components/dashboard/MainAuthenticityCard';
import { PlaceholderFeedbackModal } from '@/components/dashboard/PlaceholderFeedbackModal';
import { QuickActionCards } from '@/components/dashboard/QuickActionCards';
import { RecentAnalysesEmptyState } from '@/components/dashboard/RecentAnalysesEmptyState';
import { WelcomeSection } from '@/components/dashboard/WelcomeSection';
import { Spacing } from '@/constants/theme';

interface HomeDashboardViewProps {
  onNavigateToTab?: (tab: 'history' | 'insights' | 'settings') => void;
}

export const HomeDashboardView: React.FC<HomeDashboardViewProps> = ({
  onNavigateToTab,
}) => {
  const insets = useSafeAreaInsets();

  // State for safe placeholder action modals
  const [modalVisible, setModalVisible] = useState(false);
  const [modalTitle, setModalTitle] = useState('');
  const [modalMessage, setModalMessage] = useState('');
  const [modalTag, setModalTag] = useState('UPCOMING PHASE');

  const showPlaceholder = (title: string, message: string, tag = 'PHASE 4 PREVIEW') => {
    setModalTitle(title);
    setModalMessage(message);
    setModalTag(tag);
    setModalVisible(true);
  };

  return (
    <View style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={[
          styles.scrollContent,
          {
            paddingTop: Math.max(insets.top, 24),
            paddingBottom: Math.max(insets.bottom, 24) + 80, // Reserve space for bottom nav bar
          },
        ]}
        showsVerticalScrollIndicator={false}>
        {/* SECTION A: Compact Premium Header */}
        <DashboardHeader
          onProfilePress={() =>
            showPlaceholder(
              'User Profile & Security',
              'Profile customization and security auditing features will be available in future phases.',
              'ACCOUNT & PROFILE'
            )
          }
        />

        {/* SECTION B: Welcome Message */}
        <WelcomeSection />

        {/* SECTION C: Main Authenticity Card */}
        <MainAuthenticityCard
          onPress={() => router.push('/record')}
        />

        {/* SECTION D: Quick Action Cards (Side-by-Side) */}
        <QuickActionCards
          onRecordPress={() => router.push('/record')}
          onUploadPress={() => router.push('/upload')}
        />

        {/* SECTION E: Recent Analyses (Header + Honest Empty State) */}
        <RecentAnalysesEmptyState
          onViewAllPress={() => {
            if (onNavigateToTab) {
              onNavigateToTab('history');
            } else {
              showPlaceholder(
                'Recent Analyses History',
                'Your recorded scans will be displayed and queryable here once the storage layer is implemented.',
                'HISTORY PREVIEW'
              );
            }
          }}
        />
      </ScrollView>

      {/* Safe Action Feedback Modal */}
      <PlaceholderFeedbackModal
        visible={modalVisible}
        title={modalTitle}
        message={modalMessage}
        tag={modalTag}
        onClose={() => setModalVisible(false)}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    width: '100%',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
  },
});

export default HomeDashboardView;

