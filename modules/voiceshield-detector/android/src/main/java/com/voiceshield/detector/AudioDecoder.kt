package com.voiceshield.detector

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * Robust audio decoding engine supporting WAV, M4A, AAC, MP3, FLAC and OGG.
 * Uses MediaCodec for compressed audio and a direct RIFF parser for uncompressed PCM WAV.
 * Implements SciPy-compatible Fourier resampling to 16 kHz and strict signal validation.
 */
object AudioDecoder {

    const val TARGET_SAMPLE_RATE = 16000
    const val MAX_SAMPLES = 64000 // 4.0 seconds @ 16 kHz
    const val MIN_SAMPLES = 16000 // 1.0 second minimum @ 16 kHz
    const val SILENCE_RMS_THRESHOLD = 1e-4f // Flatline / silence floor

    data class DecodedAudio(
        val pcm: FloatArray,
        val originalSampleRate: Int,
        val originalChannels: Int,
        val durationSec: Float,
        val decodeLatencyMs: Long
    )

    /**
     * Decodes an audio file at `audioPath` to 16 kHz mono Float32 samples.
     */
    fun decode(context: Context, audioPath: String): DecodedAudio {
        val startTime = System.currentTimeMillis()
        val cleanPath = audioPath.removePrefix("file://")
        val file = File(cleanPath)

        if (!file.exists() || file.length() == 0L) {
            throw IllegalArgumentException("Audio file not found or is 0 bytes: $cleanPath")
        }

        var rawPcm: FloatArray? = null
        var origSr = TARGET_SAMPLE_RATE
        var origChannels = 1

        // 1. Fast path: Direct RIFF WAV parser
        if (file.extension.equals("wav", ignoreCase = true)) {
            val wavRes = parseWavFile(file)
            if (wavRes != null) {
                rawPcm = wavRes.pcm
                origSr = wavRes.sampleRate
                origChannels = wavRes.channels
            }
        }

        // 2. Universal path: MediaCodec for M4A, AAC, MP3, etc.
        if (rawPcm == null) {
            val codecRes = decodeWithMediaCodec(file)
            rawPcm = codecRes.pcm
            origSr = codecRes.sampleRate
            origChannels = codecRes.channels
        }

        // 3. Resample to 16 kHz if necessary using SciPy-compatible FourierResampler
        val resampledPcm = if (origSr != TARGET_SAMPLE_RATE) {
            val targetLen = (rawPcm.size.toLong() * TARGET_SAMPLE_RATE / origSr).toInt()
            FourierResampler.resample(rawPcm, targetLen)
        } else {
            rawPcm
        }

        // 4. Validate audio duration (at least 1.0s = 16,000 samples @ 16 kHz)
        if (resampledPcm.size < MIN_SAMPLES) {
            val sec = resampledPcm.size / 16000.0f
            throw IllegalArgumentException("Audio duration (${String.format("%.2f", sec)}s) is shorter than the 1.0s minimum requirement.")
        }

        // 5. Validate signal energy (silence / near-zero flatline detection)
        var sumSquares = 0.0
        for (sample in resampledPcm) {
            sumSquares += (sample * sample).toDouble()
        }
        val rms = sqrt(sumSquares / resampledPcm.size).toFloat()
        if (rms < SILENCE_RMS_THRESHOLD) {
            throw IllegalArgumentException("Audio contains only silence or near-zero ambient signal (RMS < 1e-4).")
        }

        // 6. Truncate to maximum duration (4.0s = 64,000 samples @ 16 kHz)
        val finalSamples = if (resampledPcm.size > MAX_SAMPLES) {
            resampledPcm.copyOfRange(0, MAX_SAMPLES)
        } else {
            resampledPcm
        }

        val decodeLatency = System.currentTimeMillis() - startTime
        val finalDurationSec = finalSamples.size / 16000.0f

        return DecodedAudio(
            pcm = finalSamples,
            originalSampleRate = origSr,
            originalChannels = origChannels,
            durationSec = finalDurationSec,
            decodeLatencyMs = decodeLatency
        )
    }

    private data class WavParseResult(val pcm: FloatArray, val sampleRate: Int, val channels: Int)

    private fun parseWavFile(file: File): WavParseResult? {
        return try {
            val bytes = file.readBytes()
            if (bytes.size < 44) return null
            if (bytes[0] != 'R'.code.toByte() || bytes[1] != 'I'.code.toByte() ||
                bytes[2] != 'F'.code.toByte() || bytes[3] != 'F'.code.toByte()
            ) return null

            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

            // Parse chunks
            var sampleRate = 16000
            var channels = 1
            var bitsPerSample = 16
            var pcmFloats: FloatArray? = null

            var offset = 12
            while (offset < bytes.size - 8) {
                val chunkId = String(bytes, offset, 4)
                val chunkSize = buffer.getInt(offset + 4)
                val chunkDataStart = offset + 8

                if (chunkId == "fmt ") {
                    val audioFormat = buffer.getShort(chunkDataStart).toInt()
                    channels = buffer.getShort(chunkDataStart + 2).toInt()
                    sampleRate = buffer.getInt(chunkDataStart + 4)
                    bitsPerSample = buffer.getShort(chunkDataStart + 14).toInt()

                    // Only support PCM (format 1) or IEEE Float (format 3)
                    if (audioFormat != 1 && audioFormat != 3) {
                        return null
                    }
                } else if (chunkId == "data") {
                    val numBytes = minOf(chunkSize, bytes.size - chunkDataStart)
                    if (bitsPerSample == 16) {
                        val totalShorts = numBytes / 2
                        val frames = totalShorts / channels
                        pcmFloats = FloatArray(frames)
                        val shortBuf = ByteBuffer.wrap(bytes, chunkDataStart, numBytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
                        for (i in 0 until frames) {
                            var sum = 0f
                            for (c in 0 until channels) {
                                sum += shortBuf.get() / 32768.0f
                            }
                            pcmFloats[i] = sum / channels
                        }
                    } else if (bitsPerSample == 32) {
                        val totalFloats = numBytes / 4
                        val frames = totalFloats / channels
                        pcmFloats = FloatArray(frames)
                        val floatBuf = ByteBuffer.wrap(bytes, chunkDataStart, numBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                        for (i in 0 until frames) {
                            var sum = 0f
                            for (c in 0 until channels) {
                                sum += floatBuf.get()
                            }
                            pcmFloats[i] = sum / channels
                        }
                    }
                    break
                }
                offset += 8 + chunkSize
                if (chunkSize % 2 != 0) offset += 1 // Pad byte
            }

            if (pcmFloats != null) {
                WavParseResult(pcmFloats, sampleRate, channels)
            } else {
                null
            }
        } catch (e: Exception) {
            null
        }
    }

    private data class CodecDecodeResult(val pcm: FloatArray, val sampleRate: Int, val channels: Int)

    private fun decodeWithMediaCodec(file: File): CodecDecodeResult {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(file.absolutePath)
        } catch (e: Exception) {
            throw IllegalArgumentException("Cannot open audio container: ${e.localizedMessage}")
        }

        var trackIndex = -1
        var format: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val f = extractor.getTrackFormat(i)
            val mime = f.getString(MediaFormat.KEY_MIME) ?: ""
            if (mime.startsWith("audio/")) {
                trackIndex = i
                format = f
                break
            }
        }

        if (trackIndex < 0 || format == null) {
            extractor.release()
            throw IllegalArgumentException("No audio track found in file.")
        }

        extractor.selectTrack(trackIndex)
        val mime = format.getString(MediaFormat.KEY_MIME)!
        val sampleRate = if (format.containsKey(MediaFormat.KEY_SAMPLE_RATE)) format.getInteger(MediaFormat.KEY_SAMPLE_RATE) else TARGET_SAMPLE_RATE
        val channels = if (format.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) format.getInteger(MediaFormat.KEY_CHANNEL_COUNT) else 1

        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()

        val sampleList = ArrayList<Float>(64000)
        val info = MediaCodec.BufferInfo()
        var sawInputEOS = false
        var sawOutputEOS = false

        while (!sawOutputEOS) {
            if (!sawInputEOS) {
                val inputIndex = codec.dequeueInputBuffer(10000)
                if (inputIndex >= 0) {
                    val inputBuffer = codec.getInputBuffer(inputIndex)!
                    val sampleSize = extractor.readSampleData(inputBuffer, 0)
                    if (sampleSize < 0) {
                        codec.queueInputBuffer(inputIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        sawInputEOS = true
                    } else {
                        codec.queueInputBuffer(inputIndex, 0, sampleSize, extractor.sampleTime, 0)
                        extractor.advance()
                    }
                }
            }

            val outputIndex = codec.dequeueOutputBuffer(info, 10000)
            if (outputIndex >= 0) {
                if ((info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                    sawOutputEOS = true
                }
                if (info.size > 0) {
                    val outputBuffer = codec.getOutputBuffer(outputIndex)!
                    outputBuffer.position(info.offset)
                    outputBuffer.limit(info.offset + info.size)
                    val shortBuffer = outputBuffer.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
                    while (shortBuffer.hasRemaining()) {
                        var sum = 0f
                        for (c in 0 until channels) {
                            if (shortBuffer.hasRemaining()) {
                                sum += shortBuffer.get() / 32768.0f
                            }
                        }
                        sampleList.add(sum / channels)
                    }
                }
                codec.releaseOutputBuffer(outputIndex, false)
            }
        }

        codec.stop()
        codec.release()
        extractor.release()

        val pcm = FloatArray(sampleList.size)
        for (i in sampleList.indices) {
            pcm[i] = sampleList[i]
        }
        return CodecDecodeResult(pcm, sampleRate, channels)
    }
}
