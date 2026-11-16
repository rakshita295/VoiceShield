import React, { useEffect } from 'react';
import { StyleProp, StyleSheet, View, ViewStyle } from 'react-native';
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
  ClipPath,
  Defs,
  G,
  Line,
  LinearGradient,
  Path,
  Pattern,
  Polygon,
  RadialGradient,
  Rect,
  Stop,
} from 'react-native-svg';

import { BrandColors } from '@/constants/theme';

export interface VoiceShieldLogoProps {
  /** Uniform size in logical pixels (width and height will be equal unless width/height specified) */
  size?: number;
  /** Custom width (overrides size) */
  width?: number;
  /** Custom height (overrides size) */
  height?: number;
  /** Enable subtle breathing / glow animation on the shield & halo */
  animated?: boolean;
  /** Additional container style */
  style?: StyleProp<ViewStyle>;
  /** Enable expanded wide side graphics (waveform and AI circuits) for splash/hero presentation */
  expanded?: boolean;
}

export const VoiceShieldLogo: React.FC<VoiceShieldLogoProps> = ({
  size = 220,
  width,
  height,
  animated = false,
  style,
  expanded,
}) => {
  const isExpanded = expanded ?? (width ? width > 120 : size > 120);
  const logoWidth = width ?? (isExpanded ? Math.round(size * 1.4) : size);
  const logoHeight = height ?? size;
  const currentViewBox = isExpanded ? '-60 0 420 300' : '0 0 300 300';

  // Reanimated shared values for subtle ambient glow breathing
  const pulse = useSharedValue(1);
  const haloOpacity = useSharedValue(0.75);

  useEffect(() => {
    if (animated) {
      pulse.value = withRepeat(
        withSequence(
          withTiming(1.05, { duration: 2200, easing: Easing.inOut(Easing.ease) }),
          withTiming(1.0, { duration: 2200, easing: Easing.inOut(Easing.ease) })
        ),
        -1,
        true
      );
      haloOpacity.value = withRepeat(
        withSequence(
          withTiming(1.0, { duration: 2200, easing: Easing.inOut(Easing.ease) }),
          withTiming(0.65, { duration: 2200, easing: Easing.inOut(Easing.ease) })
        ),
        -1,
        true
      );
    }
  }, [animated, pulse, haloOpacity]);

  const animatedGlowStyle = useAnimatedStyle(() => {
    if (!animated) return {};
    return {
      transform: [{ scale: pulse.value }],
      opacity: haloOpacity.value,
    };
  });

  // Shield Path definitions (viewBox 0 0 300 300)
  // Outer shield boundary
  const outerShieldPath =
    'M 150 46 C 172 46, 204 54, 216 62 C 218 116, 206 182, 150 242 C 94 182, 82 116, 84 62 C 96 54, 128 46, 150 46 Z';

  // Inner shield boundary (offset by 6px)
  const innerShieldPath =
    'M 150 54 C 168 54, 196 61, 208 68 C 210 115, 199 174, 150 230 C 101 174, 90 115, 92 68 C 104 61, 132 54, 150 54 Z';

  // Shield left facet (for 3D chiseled depth & metallic bevel)
  const leftShieldFacet =
    'M 150 54 C 132 54, 104 61, 92 68 C 90 115, 101 174, 150 230 L 150 54 Z';

  // Shield right facet
  const rightShieldFacet =
    'M 150 54 C 168 54, 196 61, 208 68 C 210 115, 199 174, 150 230 L 150 54 Z';

  return (
    <View style={[styles.container, { width: logoWidth, height: logoHeight }, style]}>
      {/* Background Animated Halo */}
      {animated && (
        <Animated.View
          style={[
            StyleSheet.absoluteFill,
            styles.haloContainer,
            animatedGlowStyle,
          ]}>
          <Svg width={logoWidth} height={logoHeight} viewBox={currentViewBox}>
            <Defs>
              <RadialGradient id="animatedHalo" cx="50%" cy="48%" r="48%">
                <Stop offset="0%" stopColor={BrandColors.electricBlue} stopOpacity="0.5" />
                <Stop offset="40%" stopColor={BrandColors.primaryPurple} stopOpacity="0.25" />
                <Stop offset="75%" stopColor={BrandColors.cyanAccent} stopOpacity="0.08" />
                <Stop offset="100%" stopColor={BrandColors.primaryBackground} stopOpacity="0" />
              </RadialGradient>
              {isExpanded && (
                <RadialGradient id="wideWingGlow" cx="50%" cy="50%" r="50%">
                  <Stop offset="0%" stopColor={BrandColors.electricBlue} stopOpacity="0.25" />
                  <Stop offset="50%" stopColor={BrandColors.primaryPurple} stopOpacity="0.12" />
                  <Stop offset="100%" stopColor={BrandColors.primaryBackground} stopOpacity="0" />
                </RadialGradient>
              )}
            </Defs>
            {isExpanded && (
              <Rect x="-50" y="55" width="400" height="178" rx="89" fill="url(#wideWingGlow)" />
            )}
            <Circle cx="150" cy="144" r="130" fill="url(#animatedHalo)" />
          </Svg>
        </Animated.View>
      )}

      {/* Main Vector Artwork */}
      <Svg width={logoWidth} height={logoHeight} viewBox={currentViewBox}>
        <Defs>
          {/* Static Circular Neon Halo Gradient */}
          <RadialGradient id="staticHalo" cx="50%" cy="48%" r="48%">
            <Stop offset="0%" stopColor={BrandColors.electricBlue} stopOpacity="0.38" />
            <Stop offset="45%" stopColor={BrandColors.primaryPurple} stopOpacity="0.2" />
            <Stop offset="75%" stopColor={BrandColors.cyanAccent} stopOpacity="0.05" />
            <Stop offset="100%" stopColor={BrandColors.primaryBackground} stopOpacity="0" />
          </RadialGradient>

          {/* Shield Glowing Border Gradient (Purple -> Electric Blue -> Cyan) */}
          <LinearGradient id="shieldBorderGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <Stop offset="0%" stopColor={BrandColors.primaryPurple} />
            <Stop offset="50%" stopColor={BrandColors.electricBlue} />
            <Stop offset="100%" stopColor={BrandColors.cyanAccent} />
          </LinearGradient>

          {/* Shield Outer Glow Stroke Gradient */}
          <LinearGradient id="shieldGlowGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <Stop offset="0%" stopColor="#C040FF" stopOpacity="0.8" />
            <Stop offset="50%" stopColor="#30A0FF" stopOpacity="0.8" />
            <Stop offset="100%" stopColor="#00E5FF" stopOpacity="0.9" />
          </LinearGradient>

          {/* Left Facet Gradient (Light Sheen) */}
          <LinearGradient id="leftFacetGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <Stop offset="0%" stopColor="#1B2348" />
            <Stop offset="50%" stopColor="#121836" />
            <Stop offset="100%" stopColor="#0A0E23" />
          </LinearGradient>

          {/* Right Facet Gradient (Deep Shadow for 3D Chisel) */}
          <LinearGradient id="rightFacetGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <Stop offset="0%" stopColor="#10152F" />
            <Stop offset="50%" stopColor="#0B0F24" />
            <Stop offset="100%" stopColor="#060918" />
          </LinearGradient>

          {/* Audio Waveform Gradient (Purple to Cyan) */}
          <LinearGradient id="waveformGrad" x1="0%" y1="50%" x2="100%" y2="50%">
            <Stop offset="0%" stopColor={BrandColors.primaryPurple} />
            <Stop offset="55%" stopColor={BrandColors.electricBlue} />
            <Stop offset="100%" stopColor={BrandColors.cyanAccent} />
          </LinearGradient>

          {/* AI Circuit Line Gradient (Electric Blue to Cyan) */}
          <LinearGradient id="circuitGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <Stop offset="0%" stopColor={BrandColors.electricBlue} />
            <Stop offset="100%" stopColor={BrandColors.cyanAccent} />
          </LinearGradient>

          {/* Padlock Shackle Metallic Gradient */}
          <LinearGradient id="shackleGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <Stop offset="0%" stopColor="#00D9FF" />
            <Stop offset="30%" stopColor="#FFFFFF" />
            <Stop offset="70%" stopColor="#168BFF" />
            <Stop offset="100%" stopColor="#00D9FF" />
          </LinearGradient>

          {/* Padlock Body High-Tech Gradient */}
          <LinearGradient id="padlockBodyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <Stop offset="0%" stopColor="#1A2247" />
            <Stop offset="50%" stopColor="#0E132D" />
            <Stop offset="100%" stopColor="#141B3B" />
          </LinearGradient>

          {/* Core Keyhole Glow */}
          <RadialGradient id="keyholeGlow" cx="50%" cy="50%" r="50%">
            <Stop offset="0%" stopColor="#FFFFFF" />
            <Stop offset="40%" stopColor={BrandColors.cyanAccent} />
            <Stop offset="100%" stopColor={BrandColors.electricBlue} />
          </RadialGradient>

          {/* Honeycomb Pattern Definition */}
          <Pattern
            id="honeycombPattern"
            width="18"
            height="31.17"
            patternUnits="userSpaceOnUse">
            {/* Hexagon 1 */}
            <Polygon
              points="9,0 18,5.19 18,15.58 9,20.78 0,15.58 0,5.19"
              fill="none"
              stroke="#00D9FF"
              strokeWidth="0.75"
              strokeOpacity="0.16"
            />
            {/* Hexagon 2 (Offset Row) */}
            <Polygon
              points="0,31.17 9,25.98 18,31.17"
              fill="none"
              stroke="#A020F0"
              strokeWidth="0.75"
              strokeOpacity="0.14"
            />
          </Pattern>

          {/* Inner Shield Clipping Path for Honeycomb */}
          <ClipPath id="innerShieldClip">
            <Path d={innerShieldPath} />
          </ClipPath>
        </Defs>

        {/* ============================================================ */}
        {/* ELEMENT 7: Circular Neon Halo & Tech Target Rings (Back Layer) */}
        {/* ============================================================ */}
        {!animated && <Circle cx="150" cy="144" r="130" fill="url(#staticHalo)" />}

        {/* Ambient Tech Calibration Rings */}
        <Circle
          cx="150"
          cy="144"
          r="124"
          stroke={BrandColors.electricBlue}
          strokeWidth="1"
          strokeDasharray="4 8"
          strokeOpacity="0.25"
          fill="none"
        />
        <Circle
          cx="150"
          cy="144"
          r="102"
          stroke={BrandColors.cyanAccent}
          strokeWidth="0.75"
          strokeDasharray="2 6"
          strokeOpacity="0.18"
          fill="none"
        />

        {/* ============================================================ */}
        {/* ELEMENT 5: Audio Waveform Extending From the Left             */}
        {/* ============================================================ */}
        {isExpanded ? (
          <G id="audioWaveformLeftExpanded">
            {/* Extended audio horizontal baseline guide */}
            <Line
              x1="-42"
              y1="144"
              x2="84"
              y2="144"
              stroke="url(#waveformGrad)"
              strokeWidth="0.8"
              strokeDasharray="3 3"
              strokeOpacity="0.45"
            />

            {/* Extended & Widened Equalizer Frequency Bars */}
            {/* Bar 1 (Far left tail) */}
            <Rect
              x="-36"
              y="138"
              width="3.5"
              height="12"
              rx="1.75"
              fill="url(#waveformGrad)"
              opacity="0.35"
            />
            {/* Bar 2 */}
            <Rect
              x="-24"
              y="134"
              width="4.5"
              height="20"
              rx="2.25"
              fill="url(#waveformGrad)"
              opacity="0.5"
            />
            {/* Bar 3 */}
            <Rect
              x="-12"
              y="126"
              width="5"
              height="36"
              rx="2.5"
              fill="url(#waveformGrad)"
              opacity="0.65"
            />
            {/* Bar 4 */}
            <Rect
              x="1"
              y="116"
              width="5.5"
              height="56"
              rx="2.75"
              fill="url(#waveformGrad)"
              opacity="0.8"
            />
            {/* Bar 5 (Major primary wave peak) */}
            <Rect
              x="15"
              y="104"
              width="6"
              height="80"
              rx="3"
              fill="url(#waveformGrad)"
              opacity="0.95"
            />
            {/* Bar 6 (Mid trough) */}
            <Rect
              x="29"
              y="122"
              width="5.5"
              height="44"
              rx="2.75"
              fill="url(#waveformGrad)"
              opacity="0.85"
            />
            {/* Bar 7 (Secondary resonance peak) */}
            <Rect
              x="42"
              y="110"
              width="6"
              height="68"
              rx="3"
              fill="url(#waveformGrad)"
              opacity="0.95"
            />
            {/* Bar 8 */}
            <Rect
              x="55"
              y="118"
              width="5.5"
              height="52"
              rx="2.75"
              fill="url(#waveformGrad)"
              opacity="0.85"
            />
            {/* Bar 9 */}
            <Rect
              x="67"
              y="128"
              width="5"
              height="32"
              rx="2.5"
              fill="url(#waveformGrad)"
              opacity="0.7"
            />
            {/* Bar 10 (Near shield flank) */}
            <Rect
              x="78"
              y="135"
              width="4"
              height="18"
              rx="2"
              fill="url(#waveformGrad)"
              opacity="0.55"
            />

            {/* Luminous Audio Frequency Pulse Dots */}
            <Circle cx="18" cy="99" r="2.5" fill={BrandColors.cyanAccent} />
            <Circle cx="45" cy="105" r="2" fill={BrandColors.electricBlue} />
            <Circle cx="3.75" cy="111" r="1.5" fill={BrandColors.cyanAccent} />
            <Circle cx="-9.5" cy="122" r="1.5" fill={BrandColors.primaryPurple} />
            <Circle cx="57.75" cy="113" r="1.5" fill={BrandColors.cyanAccent} />
          </G>
        ) : (
          <G id="audioWaveformLeft">
            {/* Subtle audio horizontal baseline guide */}
            <Line
              x1="20"
              y1="144"
              x2="88"
              y2="144"
              stroke="url(#waveformGrad)"
              strokeWidth="0.75"
              strokeDasharray="3 3"
              strokeOpacity="0.4"
            />
            {/* Equalizer Frequency Bars */}
            <Rect
              x="24"
              y="136"
              width="3"
              height="16"
              rx="1.5"
              fill="url(#waveformGrad)"
              opacity="0.45"
            />
            <Rect
              x="32"
              y="129"
              width="3.5"
              height="30"
              rx="1.75"
              fill="url(#waveformGrad)"
              opacity="0.65"
            />
            <Rect
              x="41"
              y="118"
              width="4"
              height="52"
              rx="2"
              fill="url(#waveformGrad)"
              opacity="0.8"
            />
            <Rect
              x="51"
              y="108"
              width="4.5"
              height="72"
              rx="2.25"
              fill="url(#waveformGrad)"
            />
            <Rect
              x="62"
              y="120"
              width="4"
              height="48"
              rx="2"
              fill="url(#waveformGrad)"
              opacity="0.9"
            />
            <Rect
              x="72"
              y="128"
              width="3.5"
              height="32"
              rx="1.75"
              fill="url(#waveformGrad)"
              opacity="0.75"
            />
            <Rect
              x="81"
              y="135"
              width="3"
              height="18"
              rx="1.5"
              fill="url(#waveformGrad)"
              opacity="0.6"
            />
            <Circle cx="53.25" cy="104" r="2" fill={BrandColors.cyanAccent} />
            <Circle cx="43" cy="114" r="1.5" fill={BrandColors.electricBlue} />
            <Circle cx="64" cy="116" r="1.5" fill={BrandColors.cyanAccent} />
          </G>
        )}

        {/* ============================================================ */}
        {/* ELEMENT 6: AI Circuit Lines & Glowing Nodes (Extending Right)*/}
        {/* ============================================================ */}
        {isExpanded ? (
          <G id="aiCircuitryRightExpanded">
            {/* Circuit Track 1: Upper-right diagnostic bus */}
            <Path
              d="M 215 96 L 242 96 L 264 74 L 310 74 L 326 58"
              stroke="url(#circuitGrad)"
              strokeWidth="1.75"
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            />
            {/* Terminal Node 1 */}
            <Circle cx="326" cy="58" r="5" fill={BrandColors.cyanAccent} fillOpacity="0.3" />
            <Circle cx="326" cy="58" r="2.5" fill="#FFFFFF" />
            <Circle cx="310" cy="74" r="2" fill={BrandColors.electricBlue} />

            {/* Circuit Track 2: Center horizontal main bus */}
            <Path
              d="M 217 136 L 256 136 L 276 118 L 338 118"
              stroke="url(#circuitGrad)"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            />
            {/* Terminal Node 2 (Major node) */}
            <Circle cx="338" cy="118" r="5.5" fill={BrandColors.cyanAccent} fillOpacity="0.35" />
            <Circle cx="338" cy="118" r="3" fill="#FFFFFF" />
            <Circle cx="256" cy="136" r="2.2" fill={BrandColors.cyanAccent} />

            {/* Circuit Track 3: Mid-lower neural link */}
            <Path
              d="M 256 136 L 278 158 L 322 158"
              stroke="url(#circuitGrad)"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            />
            {/* Terminal Node 3 */}
            <Circle cx="322" cy="158" r="4.5" fill={BrandColors.electricBlue} fillOpacity="0.35" />
            <Circle cx="322" cy="158" r="2.2" fill={BrandColors.cyanAccent} />

            {/* Circuit Track 4: Lower diagonal security trace */}
            <Path
              d="M 210 178 L 240 178 L 266 204 L 318 204"
              stroke="url(#circuitGrad)"
              strokeWidth="1.75"
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            />
            {/* Terminal Node 4 */}
            <Circle cx="318" cy="204" r="5" fill={BrandColors.cyanAccent} fillOpacity="0.3" />
            <Circle cx="318" cy="204" r="2.5" fill="#FFFFFF" />
            <Circle cx="240" cy="178" r="2" fill={BrandColors.electricBlue} />

            {/* Circuit Track 5: Satellite branch */}
            <Path
              d="M 266 204 L 282 220 L 302 220"
              stroke="url(#circuitGrad)"
              strokeWidth="1.3"
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            />
            <Circle cx="302" cy="220" r="2" fill={BrandColors.cyanAccent} />
          </G>
        ) : (
          <G id="aiCircuitryRight">
            {/* Circuit Track 1: Upper right diagonal branch */}
            <Path
              d="M 215 102 L 238 102 L 254 86 L 274 86"
              stroke="url(#circuitGrad)"
              strokeWidth="1.75"
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            />
            <Circle cx="274" cy="86" r="4.5" fill={BrandColors.cyanAccent} fillOpacity="0.3" />
            <Circle cx="274" cy="86" r="2.5" fill="#FFFFFF" />

            {/* Circuit Track 2: Center horizontal main bus */}
            <Path
              d="M 217 144 L 252 144 L 268 128 L 284 128"
              stroke="url(#circuitGrad)"
              strokeWidth="1.75"
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            />
            <Circle cx="284" cy="128" r="4.5" fill={BrandColors.cyanAccent} fillOpacity="0.3" />
            <Circle cx="284" cy="128" r="2.5" fill="#FFFFFF" />

            {/* Circuit Track 3: Middle branch down */}
            <Path
              d="M 242 144 L 258 160 L 276 160"
              stroke="url(#circuitGrad)"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            />
            <Circle cx="276" cy="160" r="4" fill={BrandColors.electricBlue} fillOpacity="0.35" />
            <Circle cx="276" cy="160" r="2" fill={BrandColors.cyanAccent} />

            {/* Circuit Track 4: Lower right diagonal */}
            <Path
              d="M 211 180 L 234 180 L 250 196 L 266 196"
              stroke="url(#circuitGrad)"
              strokeWidth="1.75"
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            />
            <Circle cx="266" cy="196" r="4.5" fill={BrandColors.cyanAccent} fillOpacity="0.3" />
            <Circle cx="266" cy="196" r="2.5" fill="#FFFFFF" />

            {/* Minor AI Interconnect Junction Nodes */}
            <Circle cx="238" cy="102" r="1.75" fill={BrandColors.cyanAccent} />
            <Circle cx="252" cy="144" r="1.75" fill={BrandColors.cyanAccent} />
            <Circle cx="234" cy="180" r="1.75" fill={BrandColors.electricBlue} />
          </G>
        )}

        {/* ============================================================ */}
        {/* ELEMENT 1 & 2: Large Shield with Glowing Gradient Border      */}
        {/* ============================================================ */}
        <G id="shieldBodyAndBorder">
          {/* Outer Border Glow Pass (Diffusion) */}
          <Path
            d={outerShieldPath}
            stroke="url(#shieldGlowGrad)"
            strokeWidth="7"
            strokeOpacity="0.35"
            fill="none"
          />

          {/* Secondary Glow Stroke */}
          <Path
            d={outerShieldPath}
            stroke="url(#shieldGlowGrad)"
            strokeWidth="4.5"
            strokeOpacity="0.65"
            fill="none"
          />

          {/* Crisp Primary Border Stroke */}
          <Path
            d={outerShieldPath}
            stroke="url(#shieldBorderGrad)"
            strokeWidth="2.75"
            fill="#090C1F"
          />

          {/* 3D Chiseled Shield Surface (Left Facet - Highlight) */}
          <Path d={leftShieldFacet} fill="url(#leftFacetGrad)" />

          {/* 3D Chiseled Shield Surface (Right Facet - Shadow) */}
          <Path d={rightShieldFacet} fill="url(#rightFacetGrad)" />

          {/* Central Vertical Chisel Ridge */}
          <Line
            x1="150"
            y1="54"
            x2="150"
            y2="230"
            stroke="#00D9FF"
            strokeWidth="1"
            strokeOpacity="0.35"
          />

          {/* Glossy Top & Shoulder Rim-Light Bevels */}
          <Path
            d="M 92 68 C 104 61, 132 54, 150 54"
            stroke="#FFFFFF"
            strokeWidth="1.5"
            strokeOpacity="0.45"
            strokeLinecap="round"
            fill="none"
          />
          <Path
            d="M 150 54 C 168 54, 196 61, 208 68"
            stroke={BrandColors.cyanAccent}
            strokeWidth="1.25"
            strokeOpacity="0.35"
            strokeLinecap="round"
            fill="none"
          />

          {/* ============================================================ */}
          {/* ELEMENT 3: Subtle Honeycomb Pattern Inside the Shield        */}
          {/* ============================================================ */}
          <Rect
            x="84"
            y="54"
            width="132"
            height="180"
            fill="url(#honeycombPattern)"
            clipPath="url(#innerShieldClip)"
          />
        </G>

        {/* ============================================================ */}
        {/* ELEMENT 4: Luminous Padlock at the Centre                     */}
        {/* ============================================================ */}
        <G id="luminousPadlock">
          {/* Padlock Ambient Glow */}
          <Circle cx="150" cy="144" r="26" fill="url(#keyholeGlow)" fillOpacity="0.12" />

          {/* Shackle (U-bar) */}
          <Path
            d="M 137 132 L 137 122 C 137 112, 143 106, 150 106 C 157 106, 163 112, 163 122 L 163 132"
            stroke="url(#shackleGrad)"
            strokeWidth="4.5"
            strokeLinecap="round"
            fill="none"
          />

          {/* Padlock Solid Body */}
          <Rect
            x="130"
            y="128"
            width="40"
            height="32"
            rx="6"
            fill="url(#padlockBodyGrad)"
            stroke="url(#shieldBorderGrad)"
            strokeWidth="1.75"
          />

          {/* Padlock Body Tech Inset Accent */}
          <Rect
            x="134"
            y="132"
            width="32"
            height="24"
            rx="4"
            stroke={BrandColors.cyanAccent}
            strokeWidth="0.75"
            strokeOpacity="0.4"
            fill="none"
          />

          {/* Luminous Keyhole & Cyber Core */}
          {/* Keyhole Top Bulb */}
          <Circle cx="150" cy="142" r="3.75" fill="url(#keyholeGlow)" />
          {/* Keyhole Bottom Slot */}
          <Polygon
            points="148.2,143 151.8,143 153,151 147,151"
            fill="url(#keyholeGlow)"
          />

          {/* Center Point Shimmer */}
          <Circle cx="150" cy="142" r="1.5" fill="#FFFFFF" />

          {/* Corner Cyber Accents on Padlock */}
          <Line x1="131" y1="136" x2="135" y2="136" stroke="#00D9FF" strokeWidth="1" strokeOpacity="0.8" />
          <Line x1="165" y1="136" x2="169" y2="136" stroke="#00D9FF" strokeWidth="1" strokeOpacity="0.8" />
        </G>
      </Svg>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  haloContainer: {
    justifyContent: 'center',
    alignItems: 'center',
  },
});

export default VoiceShieldLogo;
