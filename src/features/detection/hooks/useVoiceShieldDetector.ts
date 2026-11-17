import { useState, useCallback } from 'react';
import { analyzeAudioFile, VoiceShieldAnalysisResult } from 'voiceshield-detector';

export function useVoiceShieldDetector() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<VoiceShieldAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const analyze = useCallback(
    async (audioUri: string): Promise<VoiceShieldAnalysisResult | null> => {
      setIsAnalyzing(true);
      setError(null);
      try {
        const res = await analyzeAudioFile(audioUri);
        setResult(res);
        return res;
      } catch (err: any) {
        const message = err?.message || 'Deepfake voice analysis failed.';
        setError(message);
        return null;
      } finally {
        setIsAnalyzing(false);
      }
    },
    []
  );

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
    setIsAnalyzing(false);
  }, []);

  return {
    isAnalyzing,
    result,
    error,
    analyze,
    reset,
    clearError: () => setError(null),
  };
}
