package com.voiceshield.detector

import kotlin.math.*

/**
 * Fourier-domain audio resampler numerically compatible with `scipy.signal.resample`.
 *
 * Implements Bluestein's Chirp-Z FFT algorithm to support arbitrary-length (non-power-of-2)
 * signal resampling with exact Nyquist bin handling matching SciPy:
 * - When downsampling (num < N) and num is even: doubles the target Nyquist bin.
 * - When upsampling (num > N) and N is even: halves the source Nyquist bin.
 *
 * Underlying power-of-2 convolutions are computed via in-place Radix-2 Cooley-Tukey FFT.
 */
object FourierResampler {

    /**
     * Resamples a 1D real FloatArray signal to `num` samples.
     *
     * @param x Input float array of length N.
     * @param num Target length.
     * @return Resampled float array of length `num`.
     */
    fun resample(x: FloatArray, num: Int): FloatArray {
        val n = x.size
        if (n == num) return x.copyOf()
        if (num <= 0 || n <= 0) return FloatArray(0)

        // Convert input to double precision
        val xDouble = DoubleArray(n) { x[it].toDouble() }

        // Compute forward DFT via Bluestein's algorithm
        val (xReal, xImag) = bluesteinFft(xDouble, DoubleArray(n))

        val halfN = (n + 1) / 2
        val halfNum = (num + 1) / 2

        val yReal = DoubleArray(num)
        val yImag = DoubleArray(num)

        if (num > n) {
            // Upsampling: copy spectrum and zero-pad high frequencies
            for (i in 0 until halfN) {
                yReal[i] = xReal[i]
                yImag[i] = xImag[i]
            }
            if (n % 2 == 0) {
                yReal[n / 2] = xReal[n / 2] * 0.5
                yImag[n / 2] = xImag[n / 2] * 0.5
            }
            // Conjugate symmetric negative frequencies
            val numNeg = halfN - 1
            for (i in 1..numNeg) {
                yReal[num - i] = xReal[n - i]
                yImag[num - i] = xImag[n - i]
            }
        } else {
            // Downsampling: truncate high frequencies
            for (i in 0 until halfNum) {
                yReal[i] = xReal[i]
                yImag[i] = xImag[i]
            }
            if (num % 2 == 0) {
                // SciPy Nyquist bin doubling
                yReal[num / 2] = xReal[num / 2] * 2.0
                yImag[num / 2] = xImag[num / 2] * 2.0
            }
            // Copy negative frequencies
            val numNeg = halfNum - 1
            for (i in 1..numNeg) {
                yReal[num - i] = xReal[n - i]
                yImag[num - i] = xImag[n - i]
            }
        }

        // Inverse DFT: IFFT(Y) = conj(FFT(conj(Y))) / num
        for (i in 0 until num) {
            yImag[i] = -yImag[i]
        }
        val (ifftReal, _) = bluesteinFft(yReal, yImag)

        val scale = 1.0 / n.toDouble()
        val result = FloatArray(num)
        for (i in 0 until num) {
            result[i] = (ifftReal[i] * scale).toFloat()
        }
        return result
    }

    /**
     * Bluestein's FFT for arbitrary length N.
     * Converts arbitrary-size DFT into a circular convolution of size M >= 2N - 1 (power of 2).
     */
    private fun bluesteinFft(xR: DoubleArray, xI: DoubleArray): Pair<DoubleArray, DoubleArray> {
        val n = xR.size
        // Next power of 2 >= 2N - 1
        var m = 1
        val minM = 2 * n - 1
        while (m < minM) {
            m = m shl 1
        }

        // 1. Precompute chirp sequence: w[k] = exp(-j * PI * k^2 / N)
        val wR = DoubleArray(n)
        val wI = DoubleArray(n)
        for (k in 0 until n) {
            val angle = -PI * (k.toLong() * k % (2L * n)).toDouble() / n.toDouble()
            wR[k] = cos(angle)
            wI[k] = sin(angle)
        }

        // 2. Prepare array a of length M: a[k] = x[k] * w[k]
        val aR = DoubleArray(m)
        val aI = DoubleArray(m)
        for (k in 0 until n) {
            aR[k] = xR[k] * wR[k] - xI[k] * wI[k]
            aI[k] = xR[k] * wI[k] + xI[k] * wR[k]
        }

        // 3. Prepare array b of length M: b[0] = 1, b[k] = b[M - k] = conj(w[k])
        val bR = DoubleArray(m)
        val bI = DoubleArray(m)
        bR[0] = 1.0
        bI[0] = 0.0
        for (k in 1 until n) {
            val conjWR = wR[k]
            val conjWI = -wI[k]
            bR[k] = conjWR
            bI[k] = conjWI
            bR[m - k] = conjWR
            bI[m - k] = conjWI
        }

        // 4. Convolve a and b via power-of-2 FFT
        cooleyTukeyDouble(aR, aI, m, false)
        cooleyTukeyDouble(bR, bI, m, false)

        val cR = DoubleArray(m)
        val cI = DoubleArray(m)
        for (i in 0 until m) {
            cR[i] = aR[i] * bR[i] - aI[i] * bI[i]
            cI[i] = aR[i] * bI[i] + aI[i] * bR[i]
        }
        cooleyTukeyDouble(cR, cI, m, true)

        // 5. Output: X[k] = c[k] * w[k]
        val outR = DoubleArray(n)
        val outI = DoubleArray(n)
        for (k in 0 until n) {
            outR[k] = cR[k] * wR[k] - cI[k] * wI[k]
            outI[k] = cR[k] * wI[k] + cI[k] * wR[k]
        }
        return Pair(outR, outI)
    }

    /**
     * In-place Radix-2 Cooley-Tukey FFT / IFFT for power-of-2 length `n`.
     */
    private fun cooleyTukeyDouble(real: DoubleArray, imag: DoubleArray, n: Int, inverse: Boolean) {
        var j = 0
        for (i in 0 until n - 1) {
            if (i < j) {
                val tempR = real[i]; real[i] = real[j]; real[j] = tempR
                val tempI = imag[i]; imag[i] = imag[j]; imag[j] = tempI
            }
            var k = n shr 1
            while (k <= j) {
                j -= k
                k = k shr 1
            }
            j += k
        }

        var l = 2
        while (l <= n) {
            val halfL = l shr 1
            val baseAngle = (if (inverse) 2.0 else -2.0) * PI / l

            var i = 0
            while (i < n) {
                for (m in 0 until halfL) {
                    val angle = baseAngle * m
                    val wR = cos(angle)
                    val wI = sin(angle)

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
            l = l shl 1
        }

        if (inverse) {
            val invN = 1.0 / n
            for (i in 0 until n) {
                real[i] *= invN
                imag[i] *= invN
            }
        }
    }
}
