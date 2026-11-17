package com.voiceshield.detector

import kotlin.math.*

/**
 * VoiceShield LFCC Feature Extractor with Utterance-Level CMVN in pure Kotlin.
 *
 * Implements the exact mathematical specification of the VoiceShield V2 front-end:
 * 1. Reflective audio padding (256 samples on each end, matching n_fft // 2)
 * 2. 512-point STFT (win_length=400 centered inside 512, hop_length=160, periodic Hann window)
 * 3. Power spectrum (|X|^2, 257 bins)
 * 4. 20-channel linear triangular filterbank across [0, 8000 Hz]
 * 5. Log filterbank energy: ln(energy + 1e-6)
 * 6. Orthogonal Type-II DCT to retain 20 cepstral coefficients
 * 7. Delta and Delta-Delta calculation (N=2, replicate padding, denom=10.0)
 * 8. Utterance-level CMVN (per-channel zero mean, population unit variance, eps=1e-6)
 *
 * Output: Flat FloatArray formatted in row-major order for shape [1, 1, 60, numFrames]
 */
class LFCCPreprocessor(
    val sampleRate: Int = 16000,
    val nFft: Int = 512,
    val winLength: Int = 400,
    val hopLength: Int = 160,
    val nFilter: Int = 20,
    val nLfcc: Int = 20,
    val cmvnEps: Float = 1e-6f
) {
    val nFreqs: Int = nFft / 2 + 1 // 257
    val winPadLeft: Int = (nFft - winLength) / 2 // 56 samples centering offset
    val hannWindow: FloatArray = FloatArray(winLength)
    val filterbankMatrix: Array<FloatArray> // [257][20]
    val dctMatrix: Array<FloatArray>        // [20][20]

    // Precomputed twiddle factors for 512-point FFT
    private val twiddleReal: Array<FloatArray>
    private val twiddleImag: Array<FloatArray>

    init {
        // 1. Precompute periodic Hann window: 0.5 - 0.5 * cos(2 * PI * n / 400)
        for (i in 0 until winLength) {
            hannWindow[i] = (0.5 - 0.5 * cos(2.0 * Math.PI * i / winLength)).toFloat()
        }

        // 2. Precompute 20-channel linear triangular filterbank matching torchaudio
        filterbankMatrix = Array(nFreqs) { FloatArray(nFilter) }
        val allFreqs = FloatArray(nFreqs) { idx -> (idx.toFloat() * (sampleRate / 2)) / (nFreqs - 1) }
        val fPts = FloatArray(nFilter + 2) { idx -> (idx.toFloat() * (sampleRate / 2)) / (nFilter + 1) }

        for (m in 0 until nFilter) {
            val fLeft = fPts[m]
            val fCenter = fPts[m + 1]
            val fRight = fPts[m + 2]
            val fDiffLeft = fCenter - fLeft
            val fDiffRight = fRight - fCenter

            for (k in 0 until nFreqs) {
                val f = allFreqs[k]
                val downSlope = (f - fLeft) / fDiffLeft
                val upSlope = (fRight - f) / fDiffRight
                val valSlope = min(downSlope, upSlope)
                filterbankMatrix[k][m] = if (valSlope > 0.0f) valSlope else 0.0f
            }
        }

        // 3. Precompute Orthogonal DCT-II matrix (size [nFilter][nLfcc] = [20][20])
        dctMatrix = Array(nFilter) { FloatArray(nLfcc) }
        val factor = sqrt(2.0 / nFilter).toFloat()
        val c0Scale = (1.0 / sqrt(2.0)).toFloat()

        for (n in 0 until nFilter) {
            for (k in 0 until nLfcc) {
                var value = cos(Math.PI / nFilter * (n + 0.5) * k).toFloat() * factor
                if (k == 0) {
                    value *= c0Scale
                }
                dctMatrix[n][k] = value
            }
        }

        // 4. Precompute FFT twiddle factors for stages L = 2, 4, 8, ..., 512
        twiddleReal = Array(10) { FloatArray(256) }
        twiddleImag = Array(10) { FloatArray(256) }
        var stage = 1
        var l = 2
        while (l <= nFft) {
            val halfL = l shr 1
            val baseAngle = -2.0 * Math.PI / l
            for (m in 0 until halfL) {
                val angle = baseAngle * m
                twiddleReal[stage][m] = cos(angle).toFloat()
                twiddleImag[stage][m] = sin(angle).toFloat()
            }
            stage++
            l = l shl 1
        }
    }

    /**
     * Intermediate data container for comprehensive DSP numerical parity verification.
     */
    data class DetailedStages(
        val paddedAudio: FloatArray,             // [totalSamples + 512]
        val powerSpec: Array<FloatArray>,        // [257][numFrames]
        val staticLfcc: Array<FloatArray>,       // [20][numFrames]
        val delta: Array<FloatArray>,            // [20][numFrames]
        val deltaDelta: Array<FloatArray>,       // [20][numFrames]
        val cmvnFeatures: Array<FloatArray>,     // [60][numFrames]
        val finalTensor: FloatArray,             // [1 * 1 * 60 * numFrames]
        val numFrames: Int
    )

    /**
     * Extracts preprocessed LFCC+CMVN tensor for ONNX Runtime.
     *
     * @param pcmSamples 16 kHz Mono FloatArray in range [-1.0, 1.0]
     * @return Pair of: (Flat FloatArray for ONNX tensor [1, 1, 60, numFrames], numFrames)
     */
    fun extractFeatures(pcmSamples: FloatArray): Pair<FloatArray, Int> {
        val stages = extractDetailedStages(pcmSamples)
        return Pair(stages.finalTensor, stages.numFrames)
    }

    /**
     * Extracts all intermediate DSP representations alongside the final tensor.
     */
    fun extractDetailedStages(pcmSamples: FloatArray): DetailedStages {
        val totalSamples = pcmSamples.size
        val padLen = nFft / 2 // 256 samples, matching torch.stft center=True

        // 1. Reflective boundary padding: pad 256 samples on each end
        val paddedLen = totalSamples + 2 * padLen
        val paddedAudio = FloatArray(paddedLen)

        // Left reflect: pcmSamples[padLen - i]
        for (i in 0 until padLen) {
            paddedAudio[i] = pcmSamples[padLen - i]
        }
        // Center: original audio
        System.arraycopy(pcmSamples, 0, paddedAudio, padLen, totalSamples)
        // Right reflect: pcmSamples[totalSamples - 2 - i]
        for (i in 0 until padLen) {
            val idx = totalSamples - 2 - i
            paddedAudio[padLen + totalSamples + i] = if (idx >= 0) pcmSamples[idx] else pcmSamples[0]
        }

        // Calculate temporal frames: T = totalSamples / hopLength + 1
        val numFrames = (totalSamples / hopLength) + 1

        // Buffers for STFT
        val fftReal = FloatArray(nFft)
        val fftImag = FloatArray(nFft)
        val powerSpec = Array(nFreqs) { FloatArray(numFrames) }

        // 2. Static LFCC buffer: [20][numFrames]
        val staticLfcc = Array(nLfcc) { FloatArray(numFrames) }

        for (t in 0 until numFrames) {
            val frameStart = t * hopLength

            // Clear FFT buffers
            for (i in 0 until nFft) {
                fftReal[i] = 0.0f
                fftImag[i] = 0.0f
            }

            // Window is centered inside 512-point FFT buffer at offset winPadLeft = 56
            for (i in 0 until winLength) {
                val audioIdx = frameStart + winPadLeft + i
                if (audioIdx < paddedLen) {
                    fftReal[winPadLeft + i] = paddedAudio[audioIdx] * hannWindow[i]
                }
            }

            // In-place 512-point FFT
            cooleyTukeyFft(fftReal, fftImag, nFft)

            // Compute power spectrum |X|^2 for 257 frequency bins
            for (k in 0 until nFreqs) {
                powerSpec[k][t] = fftReal[k] * fftReal[k] + fftImag[k] * fftImag[k]
            }

            // Linear filterbank multiplication & natural log
            val logFilterEnergy = FloatArray(nFilter)
            for (m in 0 until nFilter) {
                var energy = 0.0f
                for (k in 0 until nFreqs) {
                    energy += powerSpec[k][t] * filterbankMatrix[k][m]
                }
                logFilterEnergy[m] = ln(energy + 1e-6f)
            }

            // Orthogonal DCT-II multiplication -> 20 static cepstral coefficients
            for (k in 0 until nLfcc) {
                var sum = 0.0f
                for (m in 0 until nFilter) {
                    sum += logFilterEnergy[m] * dctMatrix[m][k]
                }
                staticLfcc[k][t] = sum
            }
        }

        // 3. Compute Delta coefficients (N=2, replicate padding, denom=10.0)
        val delta = computeDeltas(staticLfcc, numFrames, nLfcc)

        // 4. Compute Delta-Delta coefficients (computed from delta)
        val deltaDelta = computeDeltas(delta, numFrames, nLfcc)

        // 5. Concatenate 60 channels: [60][numFrames]
        val features60 = Array(60) { c ->
            when {
                c < 20 -> staticLfcc[c]
                c < 40 -> delta[c - 20]
                else -> deltaDelta[c - 40]
            }
        }

        // 6. Utterance-Level CMVN: Zero-mean, unit-variance across time frames for each channel
        val cmvnFeatures = Array(60) { FloatArray(numFrames) }
        val outputTensor = FloatArray(1 * 1 * 60 * numFrames)

        for (c in 0 until 60) {
            val channelData = features60[c]

            // Calculate mean
            var mean = 0.0
            for (t in 0 until numFrames) {
                mean += channelData[t]
            }
            mean /= numFrames

            // Calculate population variance (unbiased = false, divisor T)
            var varSum = 0.0
            for (t in 0 until numFrames) {
                val diff = channelData[t] - mean
                varSum += diff * diff
            }
            val variance = varSum / numFrames
            val std = sqrt(variance + cmvnEps).toFloat()
            val meanF = mean.toFloat()

            // Normalize and write into row-major tensor layout: [1, 1, c, t]
            // Index in flat array: c * numFrames + t
            val channelOffset = c * numFrames
            for (t in 0 until numFrames) {
                val normVal = (channelData[t] - meanF) / std
                cmvnFeatures[c][t] = normVal
                outputTensor[channelOffset + t] = normVal
            }
        }

        return DetailedStages(
            paddedAudio = paddedAudio,
            powerSpec = powerSpec,
            staticLfcc = staticLfcc,
            delta = delta,
            deltaDelta = deltaDelta,
            cmvnFeatures = cmvnFeatures,
            finalTensor = outputTensor,
            numFrames = numFrames
        )
    }

    /**
     * Delta calculation matching torchaudio.functional.compute_deltas(mode="replicate", win_length=5).
     * Formula: d_t = (c_{t+1} - c_{t-1} + 2 * (c_{t+2} - c_{t-2})) / 10.0
     */
    private fun computeDeltas(features: Array<FloatArray>, numFrames: Int, numCoeffs: Int): Array<FloatArray> {
        val deltas = Array(numCoeffs) { FloatArray(numFrames) }
        val denom = 10.0f // 2 * sum(n^2) for n=1,2: 2 * (1 + 4) = 10

        for (c in 0 until numCoeffs) {
            val src = features[c]
            val dst = deltas[c]

            for (t in 0 until numFrames) {
                val tMinus2 = max(0, t - 2)
                val tMinus1 = max(0, t - 1)
                val tPlus1 = min(numFrames - 1, t + 1)
                val tPlus2 = min(numFrames - 1, t + 2)

                val valMinus2 = src[tMinus2]
                val valMinus1 = src[tMinus1]
                val valPlus1 = src[tPlus1]
                val valPlus2 = src[tPlus2]

                dst[t] = ((valPlus1 - valMinus1) + 2.0f * (valPlus2 - valMinus2)) / denom
            }
        }
        return deltas
    }

    /**
     * Radix-2 In-Place Cooley-Tukey FFT with precomputed twiddle factors.
     */
    private fun cooleyTukeyFft(real: FloatArray, imag: FloatArray, n: Int) {
        var j = 0
        for (i in 0 until n - 1) {
            if (i < j) {
                val tempR = real[i]
                real[i] = real[j]
                real[j] = tempR
                val tempI = imag[i]
                imag[i] = imag[j]
                imag[j] = tempI
            }
            var k = n shr 1
            while (k <= j) {
                j -= k
                k = k shr 1
            }
            j += k
        }

        var stage = 1
        var l = 2
        while (l <= n) {
            val halfL = l shr 1
            val twR = twiddleReal[stage]
            val twI = twiddleImag[stage]

            var i = 0
            while (i < n) {
                for (m in 0 until halfL) {
                    val wR = twR[m]
                    val wI = twI[m]

                    val idxA = i + m
                    val idxB = idxA + halfL
                    val uR = real[idxA]
                    val uI = imag[idxA]
                    val vR = real[idxB] * wR - imag[idxB] * wI
                    val vI = real[idxB] * wI + imag[idxB] * wR

                    real[idxA] = uR + vR
                    imag[idxA] = uI + vI
                    real[idxB] = uR - vR
                    imag[idxB] = uI - vI
                }
                i += l
            }
            stage++
            l = l shl 1
        }
    }
}
