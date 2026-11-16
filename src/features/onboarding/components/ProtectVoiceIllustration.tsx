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
} from 'react-native-svg';

import { BrandColors } from '@/constants/theme';

interface ProtectVoiceIllustrationProps {
  size?: number;
}

export const ProtectVoiceIllustration: React.FC<ProtectVoiceIllustrationProps> = ({
  size = 260,
}) => {
  // Reanimated shared values for animations
  const wavePulse1 = useSharedValue(1);
  const wavePulse2 = useSharedValue(0.7);
  const shieldScale = useSharedValue(1);
  const glowOpacity = useSharedValue(0.7);

  useEffect(() => {
    // 1. Subtle breathing on the central shield
    shieldScale.value = withRepeat(
      withSequence(
        withTiming(1.03, { duration: 2400, easing: Easing.inOut(Easing.ease) }),
        withTiming(1.0, { duration: 2400, easing: Easing.inOut(Easing.ease) })
      ),
      -1,
      true
    );

    // 2. Continuous glow oscillation
    glowOpacity.value = withRepeat(
      withSequence(
        withTiming(0.95, { duration: 2000, easing: Easing.inOut(Easing.ease) }),
        withTiming(0.55, { duration: 2000, easing: Easing.inOut(Easing.ease) })
      ),
      -1,
      true
    );

    // 3. Audio waveforms pulsing rhythmically
    wavePulse1.value = withRepeat(
      withSequence(
        withTiming(1.35, { duration: 1200, easing: Easing.inOut(Easing.cubic) }),
        withTiming(0.75, { duration: 1200, easing: Easing.inOut(Easing.cubic) })
      ),
      -1,
      true
    );

    wavePulse2.value = withRepeat(
      withSequence(
        withTiming(0.65, { duration: 1500, easing: Easing.inOut(Easing.cubic) }),
        withTiming(1.25, { duration: 1500, easing: Easing.inOut(Easing.cubic) })
      ),
      -1,
      true
    );
  }, [shieldScale, glowOpacity, wavePulse1, wavePulse2]);

  const animatedShieldStyle = useAnimatedStyle(() => ({
    transform: [{ scale: shieldScale.value }],
  }));

  const animatedGlowStyle = useAnimatedStyle(() => ({
    opacity: glowOpacity.value,
  }));

  // Shield Path coordinates (viewBox 0 0 320 280)
  const shieldOuterPath =
    'M 160 38 C 182 38, 212 46, 224 54 C 226 104, 215 168, 160 226 C 105 168, 94 104, 96 54 C 108 46, 138 38, 160 38 Z';

  const shieldLeftFacet =
    'M 160 44 C 142 44, 116 51, 104 58 C 102 102, 112 160, 160 216 L 160 44 Z';

  const shieldRightFacet =
    'M 160 44 C 178 44, 204 51, 216 58 C 218 102, 208 160, 160 216 L 160 44 Z';

  return (
    <View style={[styles.container, { width: size, height: size * 0.88 }]}>
      {/* Background Ambient Glow */}
      <Animated.View style={[StyleSheet.absoluteFill, styles.centerAlign, animatedGlowStyle]}>
        <Svg width={size} height={size * 0.88} viewBox="0 0 320 280">
          <Defs>
            <RadialGradient id="p1AmbientGlow" cx="50%" cy="48%" r="48%">
              <Stop offset="0%" stopColor={BrandColors.electricBlue} stopOpacity="0.38" />
              <Stop offset="40%" stopColor={BrandColors.primaryPurple} stopOpacity="0.22" />
              <Stop offset="75%" stopColor={BrandColors.cyanAccent} stopOpacity="0.06" />
              <Stop offset="100%" stopColor={BrandColors.primaryBackground} stopOpacity="0" />
            </RadialGradient>
          </Defs>
          <Circle cx="160" cy="132" r="120" fill="url(#p1AmbientGlow)" />
        </Svg>
      </Animated.View>

      {/* Main Vector Composition */}
      <Animated.View style={[StyleSheet.absoluteFill, styles.centerAlign, animatedShieldStyle]}>
        <Svg width={size} height={size * 0.88} viewBox="0 0 320 280">
          <Defs>
            {/* Gradients */}
            <LinearGradient id="p1ShieldBorder" x1="0%" y1="0%" x2="100%" y2="100%">
              <Stop offset="0%" stopColor={BrandColors.primaryPurple} />
              <Stop offset="50%" stopColor={BrandColors.electricBlue} />
              <Stop offset="100%" stopColor={BrandColors.cyanAccent} />
            </LinearGradient>

            <LinearGradient id="p1LeftFacet" x1="0%" y1="0%" x2="100%" y2="100%">
              <Stop offset="0%" stopColor="#1B2348" />
              <Stop offset="50%" stopColor="#121836" />
              <Stop offset="100%" stopColor="#0B0F24" />
            </LinearGradient>

            <LinearGradient id="p1RightFacet" x1="0%" y1="0%" x2="100%" y2="100%">
              <Stop offset="0%" stopColor="#121836" />
              <Stop offset="50%" stopColor="#0C1028" />
              <Stop offset="100%" stopColor="#060918" />
            </LinearGradient>

            <LinearGradient id="p1WaveLeft" x1="0%" y1="50%" x2="100%" y2="50%">
              <Stop offset="0%" stopColor={BrandColors.primaryPurple} stopOpacity="0.3" />
              <Stop offset="50%" stopColor={BrandColors.electricBlue} stopOpacity="0.75" />
              <Stop offset="100%" stopColor={BrandColors.cyanAccent} />
            </LinearGradient>

            <LinearGradient id="p1WaveRight" x1="0%" y1="50%" x2="100%" y2="50%">
              <Stop offset="0%" stopColor={BrandColors.cyanAccent} />
              <Stop offset="50%" stopColor={BrandColors.electricBlue} stopOpacity="0.75" />
              <Stop offset="100%" stopColor={BrandColors.primaryPurple} stopOpacity="0.3" />
            </LinearGradient>

            <LinearGradient id="p1MicGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <Stop offset="0%" stopColor="#FFFFFF" />
              <Stop offset="40%" stopColor={BrandColors.cyanAccent} />
              <Stop offset="100%" stopColor={BrandColors.electricBlue} />
            </LinearGradient>

            <LinearGradient id="p1MicBody" x1="0%" y1="0%" x2="100%" y2="100%">
              <Stop offset="0%" stopColor="#1E2850" />
              <Stop offset="50%" stopColor="#121A38" />
              <Stop offset="100%" stopColor="#0C122A" />
            </LinearGradient>

            <RadialGradient id="p1MicGlow" cx="50%" cy="50%" r="50%">
              <Stop offset="0%" stopColor={BrandColors.cyanAccent} stopOpacity="0.8" />
              <Stop offset="60%" stopColor={BrandColors.electricBlue} stopOpacity="0.25" />
              <Stop offset="100%" stopColor={BrandColors.electricBlue} stopOpacity="0" />
            </RadialGradient>
          </Defs>

          {/* ============================================================ */}
          {/* AUDIO WAVEFORM EXTENDING FROM LEFT OF SHIELD                 */}
          {/* ============================================================ */}
          <G id="waveLeft">
            <Line
              x1="20"
              y1="132"
              x2="94"
              y2="132"
              stroke="url(#p1WaveLeft)"
              strokeWidth="0.8"
              strokeDasharray="3 3"
              strokeOpacity="0.45"
            />
            {/* Frequency bars with varying amplitudes */}
            <Rect x="26" y="126" width="3" height="12" rx="1.5" fill="url(#p1WaveLeft)" opacity="0.4" />
            <Rect x="36" y="118" width="3.5" height="28" rx="1.75" fill="url(#p1WaveLeft)" opacity="0.6" />
            <Rect x="47" y="106" width="4" height="52" rx="2" fill="url(#p1WaveLeft)" opacity="0.85" />
            <Rect x="59" y="98" width="4.5" height="68" rx="2.25" fill="url(#p1WaveLeft)" />
            <Rect x="71" y="112" width="4" height="40" rx="2" fill="url(#p1WaveLeft)" opacity="0.9" />
            <Rect x="82" y="122" width="3.5" height="20" rx="1.75" fill="url(#p1WaveLeft)" opacity="0.65" />
            {/* Accent pulse dots */}
            <Circle cx="61.25" cy="94" r="2.25" fill={BrandColors.cyanAccent} />
            <Circle cx="49" cy="102" r="1.5" fill={BrandColors.electricBlue} />
          </G>

          {/* ============================================================ */}
          {/* AUDIO WAVEFORM EXTENDING FROM RIGHT OF SHIELD                */}
          {/* ============================================================ */}
          <G id="waveRight">
            <Line
              x1="226"
              y1="132"
              x2="300"
              y2="132"
              stroke="url(#p1WaveRight)"
              strokeWidth="0.8"
              strokeDasharray="3 3"
              strokeOpacity="0.45"
            />
            <Rect x="234" y="122" width="3.5" height="20" rx="1.75" fill="url(#p1WaveRight)" opacity="0.65" />
            <Rect x="245" y="112" width="4" height="40" rx="2" fill="url(#p1WaveRight)" opacity="0.9" />
            <Rect x="256" y="98" width="4.5" height="68" rx="2.25" fill="url(#p1WaveRight)" />
            <Rect x="269" y="106" width="4" height="52" rx="2" fill="url(#p1WaveRight)" opacity="0.85" />
            <Rect x="280" y="118" width="3.5" height="28" rx="1.75" fill="url(#p1WaveRight)" opacity="0.6" />
            <Rect x="291" y="126" width="3" height="12" rx="1.5" fill="url(#p1WaveRight)" opacity="0.4" />
            {/* Accent pulse dots */}
            <Circle cx="258.25" cy="94" r="2.25" fill={BrandColors.cyanAccent} />
            <Circle cx="271" cy="102" r="1.5" fill={BrandColors.electricBlue} />
          </G>

          {/* ============================================================ */}
          {/* CENTRAL SHIELD BODY & 3D FACETED BEVELS                      */}
          {/* ============================================================ */}
          <G id="shieldGroup">
            {/* Outer Glow Stroke */}
            <Path
              d={shieldOuterPath}
              stroke="url(#p1ShieldBorder)"
              strokeWidth="6"
              strokeOpacity="0.35"
              fill="none"
            />
            {/* Crisp Border Stroke */}
            <Path
              d={shieldOuterPath}
              stroke="url(#p1ShieldBorder)"
              strokeWidth="2.75"
              fill="#090D22"
            />

            {/* Left 3D Facet (Light Sheen) */}
            <Path d={shieldLeftFacet} fill="url(#p1LeftFacet)" />

            {/* Right 3D Facet (Shadow) */}
            <Path d={shieldRightFacet} fill="url(#p1RightFacet)" />

            {/* Vertical Center Bevel Spine */}
            <Line
              x1="160"
              y1="44"
              x2="160"
              y2="216"
              stroke={BrandColors.cyanAccent}
              strokeWidth="1"
              strokeOpacity="0.35"
            />

            {/* Top Gloss Rim Accent */}
            <Path
              d="M 104 58 C 116 51, 142 44, 160 44 C 178 44, 204 51, 216 58"
              stroke="#FFFFFF"
              strokeWidth="1.5"
              strokeOpacity="0.4"
              strokeLinecap="round"
              fill="none"
            />
          </G>

          {/* ============================================================ */}
          {/* LUMINOUS MICROPHONE SYMBOL INSIDE THE SHIELD                 */}
          {/* ============================================================ */}
          <G id="microphoneSymbol">
            {/* Microphone Ambient Radial Glow */}
            <Circle cx="160" cy="128" r="30" fill="url(#p1MicGlow)" />

            {/* Mic Outer U-Shaped Suspension Bracket */}
            <Path
              d="M 143 118 C 143 135, 177 135, 177 118"
              stroke="url(#p1MicGrad)"
              strokeWidth="3.2"
              strokeLinecap="round"
              fill="none"
            />

            {/* Vertical Mic Stem */}
            <Line
              x1="160"
              y1="135"
              x2="160"
              y2="155"
              stroke="url(#p1MicGrad)"
              strokeWidth="3.2"
              strokeLinecap="round"
            />

            {/* Weighted Mic Stand Base */}
            <Line
              x1="147"
              y1="155"
              x2="173"
              y2="155"
              stroke="url(#p1MicGrad)"
              strokeWidth="3.5"
              strokeLinecap="round"
            />

            {/* Mic Capsule Body (Rounded Pill) */}
            <Rect
              x="148"
              y="98"
              width="24"
              height="36"
              rx="12"
              fill="url(#p1MicBody)"
              stroke="url(#p1ShieldBorder)"
              strokeWidth="2"
            />

            {/* Inner Metallic Acoustic Mesh Grid */}
            <Line x1="152" y1="108" x2="168" y2="108" stroke={BrandColors.cyanAccent} strokeWidth="1" strokeOpacity="0.8" />
            <Line x1="150" y1="113" x2="170" y2="113" stroke={BrandColors.cyanAccent} strokeWidth="1" strokeOpacity="0.8" />
            <Line x1="152" y1="118" x2="168" y2="118" stroke={BrandColors.cyanAccent} strokeWidth="1" strokeOpacity="0.8" />
            <Line x1="160" y1="103" x2="160" y2="124" stroke={BrandColors.cyanAccent} strokeWidth="1" strokeOpacity="0.8" />

            {/* Audio Wave Emitting Indicator Rings from Mic Center */}
            <Circle
              cx="160"
              cy="114"
              r="2.5"
              fill="#FFFFFF"
            />
          </G>
        </Svg>
      </Animated.View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  centerAlign: {
    justifyContent: 'center',
    alignItems: 'center',
  },
});

export default ProtectVoiceIllustration;

