import React, { useMemo } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import Svg, { Defs, LinearGradient as SvgLinearGradient, Rect, Stop } from 'react-native-svg';

import { BrandColors, Spacing } from '@/constants/theme';

interface MeteringWaveformProps {
  normalizedLevel: number; // 0.0 to 1.0 based on real microphone decibels
  rawMetering?: number; // Raw decibels e.g. -24 dB
  isMeteringAvailable: boolean;
  isActive: boolean;
  isPaused?: boolean;
  height?: number;
}

const BAR_COUNT = 24;

// Preset acoustic weights for a natural voice frequency bell curve
const FREQUENCY_WEIGHTS = [
  0.22, 0.35, 0.48, 0.62, 0.78, 0.9, 0.98, 1.0, 0.96, 0.92, 0.84, 0.76,
  0.76, 0.84, 0.92, 0.96, 1.0, 0.98, 0.9, 0.78, 0.62, 0.48, 0.35, 0.22,
];

export const MeteringWaveform: React.FC<MeteringWaveformProps> = ({
  normalizedLevel,
  rawMetering,
  isMeteringAvailable,
  isActive,
  isPaused = false,
  height = 110,
}) => {
  const minBarHeight = 10;
  const maxBarHeight = height - 20;

  // Calculate each bar's dynamic height based on actual microphone level
  const barHeights = useMemo(() => {
    return FREQUENCY_WEIGHTS.map((weight, i) => {
      if (isPaused) {
        // Subdued static height when paused
        return Math.round(minBarHeight + (maxBarHeight * 0.25 * weight));
      }

      if (!isActive) {
        return minBarHeight;
      }

      // Modulate bar height with real microphone energy
      // Add subtle harmonic modulation across indices for acoustic life
      const harmonicJitter = (Math.sin(i * 1.3) * 0.15 + 1);
      const computed = minBarHeight + (maxBarHeight - minBarHeight) * normalizedLevel * weight * harmonicJitter;
      return Math.min(maxBarHeight, Math.max(minBarHeight, Math.round(computed)));
    });
  }, [normalizedLevel, isActive, isPaused, maxBarHeight]);

  const svgWidth = 320;
  const barWidth = 6;
  const barGap = 6.5;
  const totalBarsWidth = BAR_COUNT * barWidth + (BAR_COUNT - 1) * barGap;
  const startX = (svgWidth - totalBarsWidth) / 2;

  const displayDb = typeof rawMetering === 'number'
    ? `${rawMetering.toFixed(1)} dB`
    : 'Silent';

  return (
    <View style={styles.container}>
      {/* Waveform SVG */}
      <View style={[styles.waveformBox, { height }]}>
        <Svg width={svgWidth} height={height} viewBox={`0 0 ${svgWidth} ${height}`}>
          <Defs>
            <SvgLinearGradient id="liveWaveGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <Stop offset="0%" stopColor={BrandColors.cyanAccent} />
              <Stop offset="50%" stopColor={BrandColors.electricBlue} />
              <Stop offset="100%" stopColor={BrandColors.primaryPurple} />
            </SvgLinearGradient>
            <SvgLinearGradient id="pausedWaveGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <Stop offset="0%" stopColor="#4A5578" />
              <Stop offset="100%" stopColor="#222B48" />
            </SvgLinearGradient>
          </Defs>

          {barHeights.map((barHeight, idx) => {
            const x = startX + idx * (barWidth + barGap);
            const y = (height - barHeight) / 2;

            return (
              <Rect
                key={`bar-${idx}`}
                x={x}
                y={y}
                width={barWidth}
                height={barHeight}
                rx={barWidth / 2}
                fill={isPaused ? 'url(#pausedWaveGrad)' : 'url(#liveWaveGrad)'}
                opacity={isPaused ? 0.45 : Math.max(0.4, normalizedLevel * 1.1)}
              />
            );
          })}
        </Svg>
      </View>

      {/* Honest Metering Status Badge */}
      <View style={styles.meteringStatusRow}>
        <View
          style={[
            styles.meteringPulseDot,
            {
              backgroundColor: isPaused
                ? '#F59E0B'
                : isMeteringAvailable
                ? BrandColors.cyanAccent
                : BrandColors.primaryPurple,
            },
          ]}
        />
        <Text style={styles.meteringStatusText}>
          {isPaused
            ? 'AUDIO CAPTURE PAUSED • INPUT FROZEN'
            : isMeteringAvailable
            ? `LIVE MIC LEVEL • ${displayDb}`
            : 'MICROPHONE ACTIVE • REAL-TIME SAMPLING'}
        </Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
    marginVertical: Spacing.two,
  },
  waveformBox: {
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
  },
  meteringStatusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginTop: Spacing.two,
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderRadius: 20,
    backgroundColor: 'rgba(20, 26, 54, 0.75)',
    borderWidth: 1,
    borderColor: 'rgba(0, 217, 255, 0.18)',
  },
  meteringPulseDot: {
    width: 7,
    height: 7,
    borderRadius: 3.5,
  },
  meteringStatusText: {
    color: BrandColors.cyanAccent,
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 1.2,
    textTransform: 'uppercase',
  },
});

