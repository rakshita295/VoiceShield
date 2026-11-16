import { useAudioPlayer, useAudioPlayerStatus } from 'expo-audio';
import { useCallback, useEffect } from 'react';

export function useVoicePlayer(audioUri: string | null) {
  const player = useAudioPlayer(audioUri ? { uri: audioUri } : null, {
    updateInterval: 100,
  });

  const status = useAudioPlayerStatus(player);

  const togglePlayPause = useCallback(() => {
    if (!audioUri) return;
    if (status.playing) {
      player.pause();
    } else {
      // If finished, seek back to beginning before playing
      if (status.didJustFinish || (status.duration > 0 && status.currentTime >= status.duration)) {
        player.seekTo(0);
      }
      player.play();
    }
  }, [player, status, audioUri]);

  const seekTo = useCallback(
    (seconds: number) => {
      if (!audioUri) return;
      player.seekTo(seconds);
    },
    [player, audioUri]
  );

  const stop = useCallback(() => {
    if (!audioUri) return;
    player.pause();
    player.seekTo(0);
  }, [player, audioUri]);

  // Clean up audio playback when unmounting or when uri changes
  useEffect(() => {
    return () => {
      try {
        player.pause();
      } catch {
        // Safe disposal
      }
    };
  }, [player]);

  return {
    player,
    status,
    isPlaying: status.playing,
    currentTime: status.currentTime,
    duration: status.duration > 0 ? status.duration : 0,
    isLoaded: status.isLoaded,
    togglePlayPause,
    seekTo,
    stop,
  };
}

