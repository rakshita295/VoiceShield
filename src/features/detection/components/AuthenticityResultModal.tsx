import { LinearGradient } from 'expo-linear-gradient';
import React from 'react';
import {
  ActivityIndicator,
  Modal,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import Svg, { Path } from 'react-native-svg';

import { BrandColors, Spacing } from '@/constants/theme';
import { VoiceShieldAnalysisResult } from 'voiceshield-detector';

interface AuthenticityResultModalProps {
  visible: boolean;
  isLoading: boolean;
  result: VoiceShieldAnalysisResult | null;
  error: string | null;
  onClose: () => void;
}

export const AuthenticityResultModal: React.FC<AuthenticityResultModalProps> = ({
  visible,
  isLoading,
  result,
  error,
  onClose,
}) => {
  if (!visible) return null;

  const isGenuine = result?.isGenuine ?? false;

  return (
    <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
      <View style={styles.modalOverlay}>
        <View style={styles.modalContent}>
          {/* Loading State */}
          {isLoading && (
            <View style={styles.loadingContainer}>
              <View style={styles.spinnerWrapper}>
                <ActivityIndicator size="large" color={BrandColors.cyanAccent} />
              </View>
              <Text style={styles.loadingTitle}>Analyzing Voice Authenticity</Text>
              <Text style={styles.loadingSubtitle}>
                Running on-device LFCC-CMVN spectral analysis and BiLSTM neural inference off UI thread...
              </Text>
            </View>
          )}

          {/* Error State */}
          {!isLoading && error && (
            <View style={styles.stateContainer}>
              <View style={[styles.statusIconBox, { backgroundColor: 'rgba(239, 68, 68, 0.15)' }]}>
                <Svg width={32} height={32} viewBox="0 0 24 24" fill="none">
                  <Path
                    d="M12 9v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    stroke="#EF4444"
                    strokeWidth="2.2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </Svg>
              </View>
              <Text style={styles.errorTitle}>Analysis Notice</Text>
              <Text style={styles.errorMessage}>{error}</Text>
              <Pressable onPress={onClose} style={styles.dismissBtn}>
                <Text style={styles.dismissBtnText}>Dismiss</Text>
              </Pressable>
            </View>
          )}

          {/* Success Result State */}
          {!isLoading && result && (
            <View style={styles.stateContainer}>
              {/* Verdict Header Badge */}
              <View
                style={[
                  styles.statusIconBox,
                  {
                    backgroundColor: isGenuine
                      ? 'rgba(16, 185, 129, 0.15)'
                      : 'rgba(239, 68, 68, 0.18)',
                  },
                ]}>
                <Svg width={36} height={36} viewBox="0 0 24 24" fill="none">
                  {isGenuine ? (
                    <>
                      <Path
                        d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"
                        stroke="#10B981"
                        strokeWidth="2.2"
                      />
                      <Path
                        d="M9 12l2 2 4-4"
                        stroke="#10B981"
                        strokeWidth="2.2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </>
                  ) : (
                    <>
                      <Path
                        d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"
                        stroke="#EF4444"
                        strokeWidth="2.2"
                      />
                      <Path
                        d="M12 8v5M12 16h.01"
                        stroke="#EF4444"
                        strokeWidth="2.2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </>
                  )}
                </Svg>
              </View>

              <Text style={[styles.verdictTitle, { color: isGenuine ? '#10B981' : '#EF4444' }]}>
                {isGenuine ? 'GENUINE HUMAN VOICE' : 'AI-GENERATED SYNTHETIC VOICE'}
              </Text>

              <Text style={styles.verdictSubtext}>
                {isGenuine
                  ? 'Acoustic spectral features and temporal articulation match natural human speech patterns.'
                  : 'Acoustic anomalies and synthesis artifacts detected consistent with deepfake AI voice generation.'}
              </Text>

              {/* Numerical Metrics Card */}
              <View style={styles.metricsCard}>
                <View style={styles.metricRow}>
                  <Text style={styles.metricLabel}>Differential Score (s)</Text>
                  <Text style={[styles.metricValue, { color: isGenuine ? '#10B981' : '#EF4444' }]}>
                    {result.differentialScore > 0 ? `+${result.differentialScore.toFixed(3)}` : result.differentialScore.toFixed(3)}
                  </Text>
                </View>

                <View style={styles.metricRow}>
                  <Text style={styles.metricLabel}>Operating Threshold (τ)</Text>
                  <Text style={styles.metricValue}>{result.operatingThreshold.toFixed(3)}</Text>
                </View>

                <View style={styles.metricRow}>
                  <Text style={styles.metricLabel}>Threshold Margin</Text>
                  <Text style={[styles.metricValue, { color: isGenuine ? '#10B981' : '#EF4444' }]}>
                    {result.scoreMargin > 0 ? `+${result.scoreMargin.toFixed(3)}` : result.scoreMargin.toFixed(3)}
                  </Text>
                </View>

                <View style={styles.divider} />

                <View style={styles.metricRow}>
                  <Text style={styles.metricLabel}>Analyzed Duration</Text>
                  <Text style={styles.metricValue}>{result.analyzedDurationSec.toFixed(2)}s ({result.numFrames} frames)</Text>
                </View>

                <View style={styles.metricRow}>
                  <Text style={styles.metricLabel}>Total Latency</Text>
                  <Text style={styles.metricValue}>{result.latency.totalMs} ms (ONNX: {result.latency.inferenceMs} ms)</Text>
                </View>
              </View>

              {/* Privacy Footer Badge */}
              <View style={styles.privacyBadge}>
                <Svg width={14} height={14} viewBox="0 0 24 24" fill="none">
                  <Path
                    d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"
                    stroke={BrandColors.cyanAccent}
                    strokeWidth="2"
                  />
                </Svg>
                <Text style={styles.privacyBadgeText}>100% On-Device Neural Inference • Zero Cloud Uploads</Text>
              </View>

              <Pressable onPress={onClose} style={styles.actionBtn}>
                <LinearGradient
                  colors={[BrandColors.electricBlue, BrandColors.cyanAccent]}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={styles.actionGradient}>
                  <Text style={styles.actionBtnText}>Done</Text>
                </LinearGradient>
              </Pressable>
            </View>
          )}
        </View>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.78)',
    alignItems: 'center',
    justifyContent: 'center',
    padding: Spacing.four,
  },
  modalContent: {
    width: '100%',
    backgroundColor: BrandColors.cardBackground,
    borderRadius: 24,
    padding: Spacing.four,
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  loadingContainer: {
    alignItems: 'center',
    paddingVertical: Spacing.four,
    gap: 12,
  },
  spinnerWrapper: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: 'rgba(0, 217, 255, 0.1)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  loadingTitle: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '800',
    textAlign: 'center',
  },
  loadingSubtitle: {
    color: BrandColors.secondaryText,
    fontSize: 13,
    fontWeight: '500',
    textAlign: 'center',
    lineHeight: 19,
    maxWidth: 280,
  },
  stateContainer: {
    width: '100%',
    alignItems: 'center',
  },
  statusIconBox: {
    width: 68,
    height: 68,
    borderRadius: 34,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.two,
  },
  verdictTitle: {
    fontSize: 18,
    fontWeight: '900',
    letterSpacing: 0.8,
    textAlign: 'center',
    marginBottom: 6,
  },
  verdictSubtext: {
    color: BrandColors.secondaryText,
    fontSize: 12.5,
    fontWeight: '500',
    textAlign: 'center',
    lineHeight: 18,
    marginBottom: Spacing.three,
  },
  metricsCard: {
    width: '100%',
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    gap: 7,
    marginBottom: Spacing.three,
  },
  metricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  metricLabel: {
    color: BrandColors.secondaryText,
    fontSize: 12,
    fontWeight: '500',
  },
  metricValue: {
    color: '#FFFFFF',
    fontSize: 12.5,
    fontWeight: '700',
  },
  divider: {
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    marginVertical: 3,
  },
  privacyBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 20,
    backgroundColor: 'rgba(0, 217, 255, 0.08)',
    marginBottom: Spacing.three,
  },
  privacyBadgeText: {
    color: BrandColors.cyanAccent,
    fontSize: 10.5,
    fontWeight: '600',
  },
  actionBtn: {
    width: '100%',
    borderRadius: 14,
    overflow: 'hidden',
  },
  actionGradient: {
    paddingVertical: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
  actionBtnText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  errorTitle: {
    color: '#EF4444',
    fontSize: 18,
    fontWeight: '800',
    marginBottom: 8,
  },
  errorMessage: {
    color: BrandColors.secondaryText,
    fontSize: 13,
    fontWeight: '500',
    textAlign: 'center',
    lineHeight: 19,
    marginBottom: Spacing.four,
  },
  dismissBtn: {
    width: '100%',
    paddingVertical: 13,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 12,
    alignItems: 'center',
  },
  dismissBtnText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '700',
  },
});
