package com.voiceshield.detector

import android.content.Context
import ai.onnxruntime.*
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.FloatBuffer

/**
 * Expo Native Module for VoiceShield On-Device Deepfake Speech Detection.
 * Executes offline LFCC-CMVN feature extraction and ONNX inference off the UI thread.
 */
class VoiceShieldDetectorModule : Module() {
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private val preprocessor = LFCCPreprocessor()

    companion object {
        const val DECISION_THRESHOLD = -2.417969f
        const val MODEL_ASSET_PATH = "models/voiceshield_v2.onnx"
    }

    override fun definition() = ModuleDefinition {
        Name("VoiceShieldDetector")

        OnCreate {
            val context = appContext.reactContext ?: return@OnCreate
            initializeEngine(context)
        }

        OnDestroy {
            ortSession?.close()
            ortEnv?.close()
            ortSession = null
            ortEnv = null
        }

        // Off-UI Thread asynchronous analysis
        AsyncFunction("analyzeAudioFile") { audioUri: String ->
            withContext(Dispatchers.Default) {
                val context = appContext.reactContext
                    ?: throw IllegalStateException("React application context is not available.")
                analyzeAudio(context, audioUri)
            }
        }

        Function("getThreshold") {
            DECISION_THRESHOLD
        }
    }

    @Synchronized
    private fun initializeEngine(context: Context) {
        if (ortSession != null) return

        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val modelBytes = context.assets.open(MODEL_ASSET_PATH).use { it.readBytes() }
            val sessionOptions = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(2)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            ortSession = ortEnv!!.createSession(modelBytes, sessionOptions)
        } catch (e: Exception) {
            throw IllegalStateException("Failed to initialize VoiceShield ONNX engine: ${e.localizedMessage}", e)
        }
    }

    private fun analyzeAudio(context: Context, audioUri: String): Map<String, Any> {
        val totalStartTime = System.currentTimeMillis()

        // Ensure engine is initialized
        if (ortSession == null) {
            initializeEngine(context)
        }

        // 1. Audio Decoding & Validation
        val decoded = AudioDecoder.decode(context, audioUri)

        // 2. Feature Extraction (LFCC + CMVN)
        val prepStartTime = System.currentTimeMillis()
        val (featureTensor, numFrames) = preprocessor.extractFeatures(decoded.pcm)
        val prepLatency = System.currentTimeMillis() - prepStartTime

        // 3. ONNX Inference
        val inferStartTime = System.currentTimeMillis()
        val tensorShape = longArrayOf(1, 1, 60, numFrames.toLong())
        val floatBuffer = FloatBuffer.wrap(featureTensor)
        val inputTensor = OnnxTensor.createTensor(ortEnv, floatBuffer, tensorShape)

        val results = ortSession!!.run(mapOf("lfcc_features" to inputTensor))
        val logits = (results[0].value as Array<FloatArray>)[0] // [2]: [spoof, bonafide]
        val score = (results[1].value as FloatArray)[0]         // scalar differential score

        inputTensor.close()
        results.close()
        val inferLatency = System.currentTimeMillis() - inferStartTime
        val totalLatency = System.currentTimeMillis() - totalStartTime

        // 4. Decision Rule
        val isGenuine = score >= DECISION_THRESHOLD
        val spoofLogit = logits[0]
        val bonafideLogit = logits[1]

        return mapOf(
            "isGenuine" to isGenuine,
            "prediction" to if (isGenuine) "GENUINE_HUMAN" else "AI_GENERATED",
            "differentialScore" to score.toDouble(),
            "operatingThreshold" to DECISION_THRESHOLD.toDouble(),
            "scoreMargin" to (score - DECISION_THRESHOLD).toDouble(),
            "bonafideLogit" to bonafideLogit.toDouble(),
            "spoofLogit" to spoofLogit.toDouble(),
            "numFrames" to numFrames,
            "analyzedDurationSec" to decoded.durationSec.toDouble(),
            "originalSampleRate" to decoded.originalSampleRate,
            "latency" to mapOf(
                "audioDecodingMs" to decoded.decodeLatencyMs,
                "preprocessingMs" to prepLatency,
                "inferenceMs" to inferLatency,
                "totalMs" to totalLatency
            )
        )
    }
}
