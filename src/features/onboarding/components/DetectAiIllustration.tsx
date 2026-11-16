import React, { useEffect } from 'react';
import { StyleSheet, View } from 'react-native';
import Animated, {
  Easing,
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withSequence,
  withTiming,
} from 'react-native-reanimated';
import Svg, {
  Circle,
  Defs,
  G,
  Line,
  LinearGradient,
  Path,
  RadialGradient,
  Rect,
  Stop,
  Text as SvgText,
} from 'react-native-svg';

import { BrandColors } from '@/constants/theme';

interface DetectAiIllustrationProps {
  size?: number;
}

export const DetectAiIllustration: React.FC<DetectAiIllustrationProps> = ({
  size = 280,
}) => {
  // Reanimated shared values
  const scanLineX = useSharedValue(160);
  const scanGlow = useSharedValue(0.7);
  const waveHarmonic = useSharedValue(0);

  useEffect(() => {
    // 1. Smooth laser scanning line sweep across the inspection boundary (130 to 190)
    scanLineX.value = withRepeat(
      withSequence(
        withTiming(195, { duration: 2200, easing: Easing.inOut(Easing.cubic) }),
        withTiming(125, { duration: 2200, easing: Easing.inOut(Easing.cubic) })
      ),
      -1,
      true
    );

    // 2. Scanline glow intensity pulse
    scanGlow.value = withRepeat(
      withSequence(
        withTiming(1.0, { duration: 1100, easing: Easing.inOut(Easing.ease) }),
        withTiming(0.6, { duration: 1100, easing: Easing.inOut(Easing.ease) })
      ),
      -1,
      true
    );

    // 3. Independent waveform oscillation
    waveHarmonic.value = withRepeat(
      withSequence(
        withTiming(1, { duration: 1800, easing: Easing.inOut(Easing.sin) }),
        withTiming(-1, { duration: 1800, easing: Easing.inOut(Easing.sin) })
      ),
      -1,
      true
    );
  }, [scanLineX, scanGlow, waveHarmonic]);

  const animatedLaserStyle = useAnimatedStyle(() => ({
    transform: [{ translateX: scanLineX.value - 160 }],
  }));

  const animatedLaserGlow = useAnimatedStyle(() => ({
    opacity: scanGlow.value,
  }));

  return (
    <View style={[styles.container, { width: size, height: size * 0.85 }]}>
      {/* Background HUD Frame & Dual Waveforms */}
      <Svg width={size} height={size * 0.85} viewBox="0 0 320 240">
        <Defs>
          {/* Gradients */}
          <LinearGradient id="hudBorderGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <Stop offset="0%" stopColor={BrandColors.electricBlue} stopOpacity="0.8" />
            <Stop offset="50%" stopColor={BrandColors.primaryPurple} stopOpacity="0.6" />
            <Stop offset="100%" stopColor={BrandColors.cyanAccent} stopOpacity="0.8" />
          </LinearGradient>

          {/* Real Voice Waveform Gradient (Cyan / Electric Blue) */}
          <LinearGradient id="realVoiceGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <Stop offset="0%" stopColor="#168BFF" stopOpacity="0.3" />
            <Stop offset="100%" stopColor="#00D9FF" />
          </LinearGradient>

          {/* AI Voice Waveform Gradient (Purple / Magenta) */}
          <LinearGradient id="aiVoiceGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <Stop offset="0%" stopColor="#C040FF" />
            <Stop offset="100%" stopColor="#A020F0" stopOpacity="0.3" />
          </LinearGradient>

          {/* Scanning Line Glow */}
          <RadialGradient id="laserGlow" cx="50%" cy="50%" r="50%">
            <Stop offset="0%" stopColor="#FFFFFF" stopOpacity="0.9" />
            <Stop offset="40%" stopColor={BrandColors.cyanAccent} stopOpacity="0.6" />
            <Stop offset="100%" stopColor={BrandColors.electricBlue} stopOpacity="0" />
          </RadialGradient>
        </Defs>

        {/* ============================================================ */}
        {/* HUD BOUNDING CONTAINER WITH GLOWING CORNER RETICLES          */}
        {/* ============================================================ */}
        {/* Background Card */}
        <Rect
          x="16"
          y="20"
          width="288"
          height="200"
          rx="16"
          fill="#0D1230"
          stroke="rgba(22, 139, 255, 0.2)"
          strokeWidth="1.25"
        />

        {/* Grid Calibration Lines */}
        <Line x1="16" y1="120" x2="304" y2="120" stroke="rgba(169, 176, 204, 0.1)" strokeWidth="0.8" strokeDasharray="4 4" />
        <Line x1="160" y1="20" x2="160" y2="220" stroke="rgba(169, 176, 204, 0.15)" strokeWidth="1" strokeDasharray="3 3" />
        <Line x1="88" y1="20" x2="88" y2="220" stroke="rgba(169, 176, 204, 0.05)" strokeWidth="0.75" />
        <Line x1="232" y1="20" x2="232" y2="220" stroke="rgba(169, 176, 204, 0.05)" strokeWidth="0.75" />

        {/* Glowing Corner Brackets */}
        {/* Top-Left */}
        <Path d="M 22 40 L 22 26 L 36 26" stroke={BrandColors.cyanAccent} strokeWidth="2.5" fill="none" strokeLinecap="round" />
        {/* Top-Right */}
        <Path d="M 284 26 L 298 26 L 298 40" stroke={BrandColors.primaryPurple} strokeWidth="2.5" fill="none" strokeLinecap="round" />
        {/* Bottom-Left */}
        <Path d="M 22 200 L 22 214 L 36 214" stroke={BrandColors.cyanAccent} strokeWidth="2.5" fill="none" strokeLinecap="round" />
        {/* Bottom-Right */}
        <Path d="M 284 214 L 298 214 L 298 200" stroke={BrandColors.primaryPurple} strokeWidth="2.5" fill="none" strokeLinecap="round" />

        {/* ============================================================ */}
        {/* REGION 1: REAL VOICE (LEFT - ORGANIC HARMONIC WAVES)         */}
        {/* ============================================================ */}
        <G id="realVoiceRegion">
          {/* Label Badge */}
          <Rect x="28" y="32" width="90" height="22" rx="11" fill="rgba(0, 217, 255, 0.12)" stroke={BrandColors.cyanAccent} strokeWidth="1" />
          <Circle cx="39" cy="43" r="3" fill={BrandColors.cyanAccent} />
          <SvgText x="48" y="46.5" fill={BrandColors.cyanAccent} fontSize="9" fontWeight="700" letterSpacing="0.8">
            REAL VOICE
          </SvgText>

          {/* Organic Sine Curve Wave 1 (Upper Harmonic) */}
          <Path
            d="M 24 120 C 44 86, 68 86, 88 120 C 108 154, 132 154, 156 120"
            stroke="#00D9FF"
            strokeWidth="2.25"
            strokeLinecap="round"
            fill="none"
          />

          {/* Organic Sine Curve Wave 2 (Lower Harmonic) */}
          <Path
            d="M 24 120 C 50 142, 74 142, 98 120 C 122 98, 142 98, 156 120"
            stroke="#168BFF"
            strokeWidth="1.75"
            strokeOpacity="0.75"
            strokeLinecap="round"
            fill="none"
          />

          {/* Vertical Natural Spectral Frequency Bars */}
          <Rect x="36" y="108" width="3" height="24" rx="1.5" fill="url(#realVoiceGrad)" />
          <Rect x="48" y="96" width="3" height="48" rx="1.5" fill="url(#realVoiceGrad)" />
          <Rect x="60" y="88" width="3.5" height="64" rx="1.75" fill="url(#realVoiceGrad)" />
          <Rect x="72" y="102" width="3" height="36" rx="1.5" fill="url(#realVoiceGrad)" />
          <Rect x="84" y="92" width="3.5" height="56" rx="1.75" fill="url(#realVoiceGrad)" />
          <Rect x="96" y="82" width="4" height="76" rx="2" fill="url(#realVoiceGrad)" />
          <Rect x="108" y="98" width="3" height="44" rx="1.5" fill="url(#realVoiceGrad)" />
          <Rect x="120" y="90" width="3.5" height="60" rx="1.75" fill="url(#realVoiceGrad)" />
          <Rect x="132" y="104" width="3" height="32" rx="1.5" fill="url(#realVoiceGrad)" />
          <Rect x="144" y="112" width="2.5" height="16" rx="1.25" fill="url(#realVoiceGrad)" opacity="0.6" />
        </G>

        {/* ============================================================ */}
        {/* REGION 2: AI VOICE (RIGHT - QUANTIZED SYNTHETIC WAVEFORMS)  */}
        {/* ============================================================ */}
        <G id="aiVoiceRegion">
          {/* Label Badge */}
          <Rect x="202" y="32" width="90" height="22" rx="11" fill="rgba(160, 32, 240, 0.14)" stroke={BrandColors.primaryPurple} strokeWidth="1" />
          <Circle cx="213" cy="43" r="3" fill="#C040FF" />
          <SvgText x="222" y="46.5" fill="#C040FF" fontSize="9" fontWeight="700" letterSpacing="0.8">
            AI SYNTHETIC
          </SvgText>

          {/* Stepped Digital Square Pulse (AI Artifact Wave) */}
          <Path
            d="M 164 120 L 176 120 L 176 90 L 194 90 L 194 150 L 212 150 L 212 90 L 230 90 L 230 150 L 248 150 L 248 90 L 266 90 L 266 120 L 296 120"
            stroke="#C040FF"
            strokeWidth="2.25"
            strokeLinecap="square"
            fill="none"
          />

          {/* Second Quantized Stepped Harmonic */}
          <Path
            d="M 164 120 L 182 120 L 182 136 L 200 136 L 200 104 L 218 104 L 218 136 L 236 136 L 236 104 L 254 104 L 254 120 L 296 120"
            stroke="#A020F0"
            strokeWidth="1.5"
            strokeOpacity="0.75"
            strokeLinecap="square"
            fill="none"
          />

          {/* Quantized High-Tech Neural Frequency Columns */}
          <Rect x="174" y="110" width="3" height="20" rx="1.5" fill="url(#aiVoiceGrad)" opacity="0.6" />
          <Rect x="186" y="94" width="3.5" height="52" rx="1.75" fill="url(#aiVoiceGrad)" />
          <Rect x="198" y="86" width="4" height="68" rx="2" fill="url(#aiVoiceGrad)" />
          <Rect x="210" y="92" width="3.5" height="56" rx="1.75" fill="url(#aiVoiceGrad)" />
          <Rect x="222" y="84" width="4" height="72" rx="2" fill="url(#aiVoiceGrad)" />
          <Rect x="234" y="98" width="3.5" height="44" rx="1.75" fill="url(#aiVoiceGrad)" />
          <Rect x="246" y="88" width="4" height="64" rx="2" fill="url(#aiVoiceGrad)" />
          <Rect x="258" y="100" width="3.5" height="40" rx="1.75" fill="url(#aiVoiceGrad)" />
          <Rect x="270" y="112" width="3" height="16" rx="1.5" fill="url(#aiVoiceGrad)" opacity="0.6" />

          {/* Neural Synthesis Artifact Flags */}
          <Circle cx="224" cy="80" r="2.5" fill="#C040FF" />
          <Circle cx="248" cy="84" r="2" fill={BrandColors.primaryPurple} />
        </G>
      </Svg>

      {/* ============================================================ */}
      {/* VERTICAL GLOWING SCANNING LINE (TRANSLATING ACROSS INSPECTION) */}
      {/* ============================================================ */}
      <Animated.View style={[StyleSheet.absoluteFill, styles.laserLayer, animatedLaserStyle]}>
        <Animated.View style={animatedLaserGlow}>
          <Svg width={30} height={size * 0.85} viewBox="0 0 30 240">
            <Defs>
              <LinearGradient id="scanBeamGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <Stop offset="0%" stopColor="#00D9FF" stopOpacity="0" />
                <Stop offset="10%" stopColor="#00D9FF" stopOpacity="0.8" />
                <Stop offset="50%" stopColor="#FFFFFF" />
                <Stop offset="90%" stopColor="#00D9FF" stopOpacity="0.8" />
                <Stop offset="100%" stopColor="#00D9FF" stopOpacity="0" />
              </LinearGradient>
            </Defs>

            {/* Wide Laser Glow Ribbon */}
            <Rect x="11" y="24" width="8" height="192" fill="#00D9FF" opacity="0.25" />
            {/* Crisp Central Laser Beam */}
            <Line x1="15" y1="22" x2="15" y2="218" stroke="url(#scanBeamGrad)" strokeWidth="2.5" />
            {/* Top Laser Emitter Node */}
            <Circle cx="15" cy="24" r="3.5" fill="#FFFFFF" />
            <Circle cx="15" cy="24" r="5" stroke="#00D9FF" strokeWidth="1" fill="none" />
            {/* Bottom Laser Target Node */}
            <Circle cx="15" cy="216" r="3.5" fill="#FFFFFF" />
            <Circle cx="15" cy="216" r="5" stroke="#00D9FF" strokeWidth="1" fill="none" />
          </Svg>
        </Animated.View>
      </Animated.View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  laserLayer: {
    alignItems: 'center',
    justifyContent: 'center',
  },
});

export default DetectAiIllustration;

