import { LinearGradient } from 'expo-linear-gradient';
import React from 'react';
import { Modal, Pressable, StyleSheet, Text, View } from 'react-native';
import Svg, { Circle, Path } from 'react-native-svg';

import { BrandColors, Spacing } from '@/constants/theme';

interface PlaceholderFeedbackModalProps {
  visible: boolean;
  title: string;
  message: string;
  tag?: string;
  onClose: () => void;
}

export const PlaceholderFeedbackModal: React.FC<PlaceholderFeedbackModalProps> = ({
  visible,
  title,
  message,
  tag = 'UPCOMING PHASE',
  onClose,
}) => {
  return (
    <Modal
      visible={visible}
      transparent
      animationType="fade"
      onRequestClose={onClose}>
      <Pressable style={styles.backdrop} onPress={onClose}>
        <Pressable style={styles.modalCard} onPress={(e) => e.stopPropagation()}>
          {/* Top Tag */}
          <View style={styles.tagBadge}>
            <View style={styles.tagDot} />
            <Text style={styles.tagText}>{tag}</Text>
          </View>

          {/* Icon */}
          <View style={styles.iconCircle}>
            <Svg width={32} height={32} viewBox="0 0 24 24" fill="none">
              <Circle cx="12" cy="12" r="10" stroke={BrandColors.cyanAccent} strokeWidth="2" />
              <Path
                d="M 12 8 L 12 12 M 12 16 L 12.01 16"
                stroke={BrandColors.cyanAccent}
                strokeWidth="2.5"
                strokeLinecap="round"
              />
            </Svg>
          </View>

          {/* Title & Message */}
          <Text style={styles.titleText}>{title}</Text>
          <Text style={styles.messageText}>{message}</Text>

          {/* Dismiss Button */}
          <Pressable
            onPress={onClose}
            style={({ pressed }) => [
              styles.dismissButton,
              pressed && styles.dismissButtonPressed,
            ]}>
            <LinearGradient
              colors={[BrandColors.primaryPurple, BrandColors.electricBlue]}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={styles.buttonGradient}>
              <Text style={styles.buttonText}>Got It</Text>
            </LinearGradient>
          </Pressable>
        </Pressable>
      </Pressable>
    </Modal>
  );
};

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: 'rgba(4, 5, 14, 0.75)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: Spacing.four,
  },
  modalCard: {
    width: '100%',
    maxWidth: 340,
    backgroundColor: '#141A36',
    borderRadius: 24,
    borderWidth: 1.25,
    borderColor: 'rgba(22, 139, 255, 0.35)',
    padding: Spacing.four,
    alignItems: 'center',
    shadowColor: BrandColors.electricBlue,
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.35,
    shadowRadius: 16,
    elevation: 8,
  },
  tagBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
    backgroundColor: 'rgba(22, 139, 255, 0.1)',
    borderWidth: 1,
    borderColor: 'rgba(0, 217, 255, 0.25)',
    marginBottom: Spacing.three,
  },
  tagDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: BrandColors.cyanAccent,
  },
  tagText: {
    color: BrandColors.cyanAccent,
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 1.2,
  },
  iconCircle: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: 'rgba(0, 217, 255, 0.08)',
    borderWidth: 1,
    borderColor: 'rgba(0, 217, 255, 0.3)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.three,
  },
  titleText: {
    color: '#FFFFFF',
    fontSize: 19,
    fontWeight: '800',
    textAlign: 'center',
    marginBottom: Spacing.two,
  },
  messageText: {
    color: BrandColors.secondaryText,
    fontSize: 13.5,
    lineHeight: 20,
    textAlign: 'center',
    marginBottom: Spacing.four,
  },
  dismissButton: {
    width: '100%',
    borderRadius: 24,
    overflow: 'hidden',
  },
  dismissButtonPressed: {
    opacity: 0.85,
  },
  buttonGradient: {
    paddingVertical: 12,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 24,
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '700',
    letterSpacing: 0.4,
  },
});

export default PlaceholderFeedbackModal;

