import React, { useEffect } from 'react';
import { Dimensions, StyleSheet, View } from 'react-native';
import Animated, {
  Easing,
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withTiming,
} from 'react-native-reanimated';
import Svg, {
  Defs,
  LinearGradient,
  Path,
  Rect,
  Stop,
} from 'react-native-svg';

import { BrandColors } from '@/constants/theme';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

interface FlowingWaveformProps {
  height?: number;
}

export const FlowingWaveform: React.FC<FlowingWaveformProps> = ({ height = 90 }) => {
  const waveOffset = useSharedValue(0);
  const barPulse = useSharedValue(1);

  useEffect(() => {
    // Continuous smooth horizontal wave progression
    waveOffset.value = withRepeat(
      withTiming(-SCREEN_WIDTH, {
        duration: 7000,
        easing: Easing.linear,
      }),
      -1,
      false
    );

    // Subtle rhythmic audio breathing on the equalizer bars
    barPulse.value = withRepeat(
      withTiming(1.3, {
        duration: 1800,
        easing: Easing.inOut(Easing.ease),
      }),
      -1,
      true
    );
  }, [waveOffset, barPulse]);

  const animatedWaveStyle1 = useAnimatedStyle(() => ({
    transform: [{ translateX: waveOffset.value }],
  }));

  const animatedWaveStyle2 = useAnimatedStyle(() => ({
    transform: [{ translateX: waveOffset.value * 0.7 }],
  }));

  // Create a continuous seamless path spanning 2 full screen widths
  const waveWidth = Math.max(SCREEN_WIDTH, 400);
  const pathD = `
    M 0 45
    Q ${waveWidth * 0.125} 15, ${waveWidth * 0.25} 45
    T ${waveWidth * 0.5} 45
    T ${waveWidth * 0.75} 45
    T ${waveWidth} 45
    T ${waveWidth * 1.25} 45
    T ${waveWidth * 1.5} 45
    T ${waveWidth * 1.75} 45
    T ${waveWidth * 2} 45
    L ${waveWidth * 2} ${height}
    L 0 ${height}
    Z
  `;

  const secondaryPathD = `
    M 0 52
    Q ${waveWidth * 0.15} 25, ${waveWidth * 0.3} 52
    T ${waveWidth * 0.6} 52
    T ${waveWidth * 0.9} 52
    T ${waveWidth * 1.2} 52
    T ${waveWidth * 1.5} 52
    T ${waveWidth * 1.8} 52
    T ${waveWidth * 2.1} 52
    L ${waveWidth * 2.1} ${height}
    L 0 ${height}
    Z
  `;

  // Array of delicate audio frequency bars in the center
  const numBars = 24;
  const barWidth = 3;
  const barSpacing = 9;
  const totalBarsWidth = numBars * (barWidth + barSpacing);
  const startX = (SCREEN_WIDTH - totalBarsWidth) / 2;

  // Heights array with organic audio distribution
  const barHeights = [
    8, 12, 18, 26, 34, 42, 50, 44, 36, 28, 20, 32, 46, 54, 48, 38, 26, 18, 28, 36, 24, 16, 10, 6,
  ];

  return (
    <View style={[styles.container, { height }]}>
      {/* Background Soft Waveform (Secondary harmonic) */}
      <Animated.View style={[styles.waveLayer, animatedWaveStyle2]}>
        <Svg width={waveWidth * 2.1} height={height}>
          <Defs>
            <LinearGradient id="secWaveGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <Stop offset="0%" stopColor={BrandColors.primaryPurple} stopOpacity="0.18" />
              <Stop offset="50%" stopColor={BrandColors.electricBlue} stopOpacity="0.25" />
              <Stop offset="100%" stopColor={BrandColors.cyanAccent} stopOpacity="0.18" />
            </LinearGradient>
          </Defs>
          <Path d={secondaryPathD} fill="url(#secWaveGrad)" />
        </Svg>
      </Animated.View>

      {/* Primary Flowing Waveform */}
      <Animated.View style={[styles.waveLayer, animatedWaveStyle1]}>
        <Svg width={waveWidth * 2} height={height}>
          <Defs>
            <LinearGradient id="primaryWaveGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <Stop offset="0%" stopColor={BrandColors.primaryPurple} stopOpacity="0.3" />
              <Stop offset="35%" stopColor={BrandColors.electricBlue} stopOpacity="0.45" />
              <Stop offset="70%" stopColor={BrandColors.cyanAccent} stopOpacity="0.4" />
              <Stop offset="100%" stopColor={BrandColors.primaryPurple} stopOpacity="0.3" />
            </LinearGradient>
            <LinearGradient id="waveStrokeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <Stop offset="0%" stopColor={BrandColors.primaryPurple} stopOpacity="0.8" />
              <Stop offset="40%" stopColor={BrandColors.electricBlue} />
              <Stop offset="80%" stopColor={BrandColors.cyanAccent} />
              <Stop offset="100%" stopColor={BrandColors.primaryPurple} stopOpacity="0.8" />
            </LinearGradient>
          </Defs>
          <Path d={pathD} fill="url(#primaryWaveGrad)" />
          {/* Surface crest glow line */}
          <Path
            d={pathD.split('L')[0]}
            fill="none"
            stroke="url(#waveStrokeGrad)"
            strokeWidth="1.5"
            strokeOpacity="0.75"
          />
        </Svg>
      </Animated.View>

      {/* Centered Rhythmic Equalizer Baseline Accent */}
      <View style={styles.equalizerOverlay}>
        <Svg width={SCREEN_WIDTH} height={50}>
          <Defs>
            <LinearGradient id="eqBarGrad" x1="0%" y1="100%" x2="0%" y2="0%">
              <Stop offset="0%" stopColor={BrandColors.electricBlue} stopOpacity="0.2" />
              <Stop offset="60%" stopColor={BrandColors.cyanAccent} stopOpacity="0.75" />
              <Stop offset="100%" stopColor="#FFFFFF" stopOpacity="0.95" />
            </LinearGradient>
          </Defs>
          {barHeights.map((h, i) => {
            const bx = startX + i * (barWidth + barSpacing);
            const by = 50 - h * 0.7;
            return (
              <Rect
                key={i}
                x={bx}
                y={by}
                width={barWidth}
                height={h * 0.7}
                rx={1.5}
                fill="url(#eqBarGrad)"
              />
            );
          })}
        </Svg>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '100%',
    overflow: 'hidden',
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    justifyContent: 'flex-end',
  },
  waveLayer: {
    position: 'absolute',
    bottom: 0,
    left: 0,
  },
  equalizerOverlay: {
    position: 'absolute',
    bottom: 12,
    left: 0,
    right: 0,
    alignItems: 'center',
    justifyContent: 'center',
  },
});

export default FlowingWaveform;

