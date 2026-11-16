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
  Ellipse,
  G,
  Line,
  LinearGradient,
  Path,
  Polygon,
  RadialGradient,
  Rect,
  Stop,
  Text as SvgText,
} from 'react-native-svg';

import { BrandColors } from '@/constants/theme';

interface PrivacyMattersIllustrationProps {
  size?: number;
}

export const PrivacyMattersIllustration: React.FC<PrivacyMattersIllustrationProps> = ({
  size = 270,
}) => {
  // Reanimated shared values
  const floatY = useSharedValue(0);
  const orbitAngle = useSharedValue(0);
  const lockGlow = useSharedValue(0.7);

  useEffect(() => {
    // 1. Gentle floating effect on the shield (smooth hovering)
    floatY.value = withRepeat(
      withSequence(
        withTiming(-6, { duration: 2200, easing: Easing.inOut(Easing.sin) }),
        withTiming(6, { duration: 2200, easing: Easing.inOut(Easing.sin) })
      ),
      -1,
      true
    );

    // 2. Slow continuous orbit rotation
    orbitAngle.value = withRepeat(
      withTiming(360, { duration: 16000, easing: Easing.linear }),
      -1,
      false
    );

    // 3. Subtle pulsing lock glow
    lockGlow.value = withRepeat(
      withSequence(
        withTiming(1.0, { duration: 1500, easing: Easing.inOut(Easing.ease) }),
        withTiming(0.6, { duration: 1500, easing: Easing.inOut(Easing.ease) })
      ),
      -1,
      true
    );
  }, [floatY, orbitAngle, lockGlow]);

  const animatedShieldFloating = useAnimatedStyle(() => ({
    transform: [{ translateY: floatY.value }],
  }));

  const animatedOrbitStyle = useAnimatedStyle(() => ({
    transform: [{ rotateZ: `${orbitAngle.value}deg` }],
  }));

  const animatedLockGlow = useAnimatedStyle(() => ({
    opacity: lockGlow.value,
  }));

  // Shield Path coordinates (viewBox 0 0 320 280)
  const shieldOuterPath =
    'M 160 42 C 182 42, 212 50, 224 58 C 226 108, 215 172, 160 230 C 105 172, 94 108, 96 58 C 108 50, 138 42, 160 42 Z';

  const shieldLeftFacet =
    'M 160 48 C 142 48, 116 55, 104 62 C 102 106, 112 164, 160 220 L 160 48 Z';

  const shieldRightFacet =
    'M 160 48 C 178 48, 204 55, 216 62 C 218 106, 208 164, 160 220 L 160 48 Z';

  return (
    <View style={[styles.container, { width: size, height: size * 0.92 }]}>
      {/* Background Radial Glow */}
      <Svg width={size} height={size * 0.92} viewBox="0 0 320 280" style={StyleSheet.absoluteFill}>
        <Defs>
          <RadialGradient id="p3AmbientGlow" cx="50%" cy="48%" r="48%">
            <Stop offset="0%" stopColor={BrandColors.electricBlue} stopOpacity="0.35" />
            <Stop offset="40%" stopColor={BrandColors.primaryPurple} stopOpacity="0.22" />
            <Stop offset="75%" stopColor={BrandColors.cyanAccent} stopOpacity="0.06" />
            <Stop offset="100%" stopColor={BrandColors.primaryBackground} stopOpacity="0" />
          </RadialGradient>
        </Defs>
        <Circle cx="160" cy="136" r="125" fill="url(#p3AmbientGlow)" />
      </Svg>

      {/* Rotating Cyber Orbit Ring Layer */}
      <Animated.View style={[StyleSheet.absoluteFill, styles.centerAlign, animatedOrbitStyle]}>
        <Svg width={size} height={size * 0.92} viewBox="0 0 320 280">
          <Defs>
            <LinearGradient id="orbitGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
              <Stop offset="0%" stopColor={BrandColors.primaryPurple} stopOpacity="0.8" />
              <Stop offset="50%" stopColor={BrandColors.electricBlue} stopOpacity="0.4" />
              <Stop offset="100%" stopColor={BrandColors.cyanAccent} stopOpacity="0.9" />
            </LinearGradient>
          </Defs>

          {/* Elliptical Orbital Track 1 */}
          <Ellipse
            cx="160"
            cy="136"
            rx="132"
            ry="64"
            stroke="url(#orbitGrad1)"
            strokeWidth="1.5"
            strokeDasharray="6 8"
            fill="none"
          />

          {/* Orbital Satellite Node 1 (Cyan) */}
          <Circle cx="292" cy="136" r="4.5" fill={BrandColors.cyanAccent} />
          <Circle cx="292" cy="136" r="7" stroke={BrandColors.cyanAccent} strokeWidth="1" strokeOpacity="0.5" fill="none" />

          {/* Orbital Satellite Node 2 (Purple) */}
          <Circle cx="28" cy="136" r="4" fill="#C040FF" />
          <Circle cx="28" cy="136" r="6" stroke="#C040FF" strokeWidth="1" strokeOpacity="0.5" fill="none" />
        </Svg>
      </Animated.View>

      {/* Floating Shield and Central Luminous Lock */}
      <Animated.View style={[StyleSheet.absoluteFill, styles.centerAlign, animatedShieldFloating]}>
        <Svg width={size} height={size * 0.92} viewBox="0 0 320 280">
          <Defs>
            <LinearGradient id="p3ShieldBorder" x1="0%" y1="0%" x2="100%" y2="100%">
              <Stop offset="0%" stopColor={BrandColors.primaryPurple} />
              <Stop offset="50%" stopColor={BrandColors.electricBlue} />
              <Stop offset="100%" stopColor={BrandColors.cyanAccent} />
            </LinearGradient>

            <LinearGradient id="p3LeftFacet" x1="0%" y1="0%" x2="100%" y2="100%">
              <Stop offset="0%" stopColor="#1C244A" />
              <Stop offset="50%" stopColor="#121838" />
              <Stop offset="100%" stopColor="#0B0F24" />
            </LinearGradient>

            <LinearGradient id="p3RightFacet" x1="0%" y1="0%" x2="100%" y2="100%">
              <Stop offset="0%" stopColor="#121838" />
              <Stop offset="50%" stopColor="#0C1028" />
              <Stop offset="100%" stopColor="#060918" />
            </LinearGradient>

            <LinearGradient id="p3ShackleGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <Stop offset="0%" stopColor="#00D9FF" />
              <Stop offset="30%" stopColor="#FFFFFF" />
              <Stop offset="70%" stopColor="#168BFF" />
              <Stop offset="100%" stopColor="#00D9FF" />
            </LinearGradient>

            <LinearGradient id="p3LockBody" x1="0%" y1="0%" x2="100%" y2="100%">
              <Stop offset="0%" stopColor="#1B244C" />
              <Stop offset="50%" stopColor="#0E1430" />
              <Stop offset="100%" stopColor="#141B3E" />
            </LinearGradient>

            <RadialGradient id="p3KeyholeGlow" cx="50%" cy="50%" r="50%">
              <Stop offset="0%" stopColor="#FFFFFF" />
              <Stop offset="40%" stopColor={BrandColors.cyanAccent} />
              <Stop offset="100%" stopColor={BrandColors.electricBlue} />
            </RadialGradient>
          </Defs>

          {/* ============================================================ */}
          {/* CENTRAL SHIELD GEOMETRY                                      */}
          {/* ============================================================ */}
          <G id="shieldLayer">
            {/* Outer Glow Stroke */}
            <Path
              d={shieldOuterPath}
              stroke="url(#p3ShieldBorder)"
              strokeWidth="6"
              strokeOpacity="0.35"
              fill="none"
            />
            {/* Crisp Border Stroke */}
            <Path
              d={shieldOuterPath}
              stroke="url(#p3ShieldBorder)"
              strokeWidth="2.75"
              fill="#090D22"
            />

            {/* Left 3D Facet (Light Sheen) */}
            <Path d={shieldLeftFacet} fill="url(#p3LeftFacet)" />

            {/* Right 3D Facet (Shadow) */}
            <Path d={shieldRightFacet} fill="url(#p3RightFacet)" />

            {/* Vertical Center Bevel Spine */}
            <Line
              x1="160"
              y1="48"
              x2="160"
              y2="220"
              stroke={BrandColors.cyanAccent}
              strokeWidth="1"
              strokeOpacity="0.35"
            />

            {/* Top Gloss Rim Accent */}
            <Path
              d="M 104 62 C 116 55, 142 48, 160 48 C 178 48, 204 55, 216 62"
              stroke="#FFFFFF"
              strokeWidth="1.5"
              strokeOpacity="0.4"
              strokeLinecap="round"
              fill="none"
            />
          </G>

          {/* ============================================================ */}
          {/* LUMINOUS CYBER LOCK AT CENTER                                */}
          {/* ============================================================ */}
          <G id="cyberLock">
            {/* Ambient Keyhole Glow */}
            <Circle cx="160" cy="136" r="28" fill="url(#p3KeyholeGlow)" fillOpacity="0.14" />

            {/* Shackle U-Bar */}
            <Path
              d="M 145 124 L 145 112 C 145 102, 151 96, 160 96 C 169 96, 175 102, 175 112 L 175 124"
              stroke="url(#p3ShackleGrad)"
              strokeWidth="4.5"
              strokeLinecap="round"
              fill="none"
            />

            {/* Padlock Solid Body */}
            <Rect
              x="138"
              y="120"
              width="44"
              height="36"
              rx="7"
              fill="url(#p3LockBody)"
              stroke="url(#p3ShieldBorder)"
              strokeWidth="1.75"
            />

            {/* Padlock Inner Bevel */}
            <Rect
              x="142"
              y="124"
              width="36"
              height="28"
              rx="4"
              stroke={BrandColors.cyanAccent}
              strokeWidth="0.8"
              strokeOpacity="0.4"
              fill="none"
            />

            {/* Luminous Keyhole Core */}
            <Circle cx="160" cy="134" r="4.25" fill="url(#p3KeyholeGlow)" />
            <Polygon
              points="158,135 162,135 163.5,144 156.5,144"
              fill="url(#p3KeyholeGlow)"
            />
            {/* Spark Reflection */}
            <Circle cx="160" cy="134" r="1.5" fill="#FFFFFF" />

            {/* Decorative Cyber Hash Lines on Lock */}
            <Line x1="140" y1="128" x2="144" y2="128" stroke="#00D9FF" strokeWidth="1" strokeOpacity="0.75" />
            <Line x1="176" y1="128" x2="180" y2="128" stroke="#00D9FF" strokeWidth="1" strokeOpacity="0.75" />
          </G>

          {/* ============================================================ */}
          {/* SMALL PRIVACY / ON-DEVICE SECURITY INDICATOR BADGE           */}
          {/* ============================================================ */}
          <G id="privacyBadge">
            <Rect
              x="88"
              y="244"
              width="144"
              height="24"
              rx="12"
              fill="rgba(13, 18, 48, 0.9)"
              stroke={BrandColors.cyanAccent}
              strokeWidth="1"
              strokeOpacity="0.35"
            />
            {/* Pulsing Green/Cyan Privacy Dot */}
            <Circle cx="102" cy="256" r="3.5" fill={BrandColors.cyanAccent} />
            <Circle cx="102" cy="256" r="6" stroke={BrandColors.cyanAccent} strokeWidth="1" strokeOpacity="0.4" fill="none" />
            {/* Badge Text */}
            <SvgText
              x="114"
              y="259.5"
              fill="#FFFFFF"
              fontSize="9"
              fontWeight="700"
              letterSpacing="1">
              SECURE ON-DEVICE
            </SvgText>
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

export default PrivacyMattersIllustration;

