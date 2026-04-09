package com.gabriel.meterreader

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.os.Bundle
import android.os.Environment
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.View
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.gabriel.meterreader.databinding.ActivityMainBinding
import com.gabriel.meterreader.domain.Detection
import com.gabriel.meterreader.domain.ReaderState
import com.gabriel.meterreader.ml.ModelConfig
import com.gabriel.meterreader.ml.ReadingLogic
import com.gabriel.meterreader.ml.OnnxYoloDetector
import com.gabriel.meterreader.util.BitmapUtils
import com.gabriel.meterreader.util.LetterboxResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.abs

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService

    private var displayDetector: OnnxYoloDetector? = null
    private var digitDetector: OnnxYoloDetector? = null

    private val isProcessing = AtomicBoolean(false)

    private var currentState: ReaderState = ReaderState.PATROL
    private var lastPatrolInferenceAt = 0L
    private val patrolIntervalMs = PATROL_INTERVAL_MS
    private var lastLiveDigitsInferenceAt = 0L
    private val liveDigitsIntervalMs = LIVE_DIGIT_INTERVAL_MS

    private var lastStableBox: RectF? = null
    private var stableSinceMs = 0L
    private val stabilizationThresholdMs = 800L

    private var previewFrameWidth: Int = 1
    private var previewFrameHeight: Int = 1
    private var finalReading: String? = null
    private var lastLiveDisplayBox: RectF? = null
    private var lastLiveDisplayConfidence: Float = 0f
    private var lastLiveDigits: List<Detection> = emptyList()
    private var lastLiveReading: String? = null
    private var lastLiveReadingConfidence: Float? = null
    private var lastLiveSeenAtMs: Long = 0L

    // --- Mode: live vs photo ---
    private var isPhotoMode = false
    private val captureNextFrame = AtomicBoolean(false)
    private var isDigitOnlyMode = false

    companion object {
        private const val TAG = "MainActivity"
        /** Display panels are typically 3.5× wider than tall. */
        private const val DISPLAY_ASPECT_RATIO = 3.5f
        /** Fast live scanning for field usage; tune on target hardware if needed. */
        private const val PATROL_INTERVAL_MS = 120L
        private const val LIVE_DIGIT_INTERVAL_MS = 180L
        /** In digits-only mode we assume the meter display is centered and almost full width. */
        private const val CENTRAL_ROI_WIDTH_RATIO = 0.92f
        private const val DIGIT_ONLY_SYNTHETIC_CONFIDENCE = 1f
        private const val LIVE_OVERLAY_HOLD_MS = 900L
        private const val MAX_CAPTURE_EDGE = 1600
        private const val MAX_CAPTURE_FILES = 200
    }

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startCamera()
            } else {
                updateUi(
                    state = ReaderState.ERROR,
                    reading = null,
                    message = "Permiso de cámara denegado."
                )
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()
        setupButtons()

        lifecycleScope.launch {
            initializeModels()
            requestCameraOrStart()
        }
    }

    private suspend fun initializeModels() = withContext(Dispatchers.IO) {
        try {
            displayDetector = OnnxYoloDetector(
                context = applicationContext,
                config = ModelConfig(
                    modelFile = "display_detection_int8.onnx",
                    labelsFile = "labels_display.txt",
                    confidenceThreshold = 0.25f,
                    iouThreshold = 0.50f
                )
            )
            digitDetector = OnnxYoloDetector(
                context = applicationContext,
                config = ModelConfig(
                    modelFile = "digit_recognition_int8.onnx",
                    labelsFile = "labels_digits.txt",
                    confidenceThreshold = 0.30f,
                    iouThreshold = 0.20f
                )
            )
            withContext(Dispatchers.Main) {
                updateUi(
                    state = ReaderState.PATROL,
                    reading = null,
                    message = "Apunta al medidor. Análisis en vivo activado."
                )
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Model init error", e)
            withContext(Dispatchers.Main) {
                updateUi(
                    state = ReaderState.ERROR,
                    reading = null,
                    message = "No se pudieron cargar los modelos en assets."
                )
            }
        }
    }

    private fun requestCameraOrStart() {
        when {
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED -> startCamera()

            else -> cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val cameraProvider = providerFuture.get()

            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .build()
                .also { it.surfaceProvider = binding.previewView.surfaceProvider }

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(1280, 720))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        analyzeImage(imageProxy)
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeImage(imageProxy: ImageProxy) {
        try {
            if (currentState == ReaderState.PROCESSING || (isPhotoMode && currentState == ReaderState.READY)) {
                imageProxy.close()
                return
            }

            val now = SystemClock.elapsedRealtime()

            if (isPhotoMode) {
                // In photo mode only process the frame the user explicitly triggered.
                if (!captureNextFrame.compareAndSet(true, false)) {
                    imageProxy.close()
                    return
                }
            } else {
                if (now - lastPatrolInferenceAt < patrolIntervalMs) {
                    imageProxy.close()
                    return
                }
            }

            if (!isProcessing.compareAndSet(false, true)) {
                imageProxy.close()
                return
            }

            if (!isPhotoMode) lastPatrolInferenceAt = now

            val rawBitmap = imageProxy.toBitmap()
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
            imageProxy.close()

            // Rotate the bitmap to match the display orientation so that YOLO receives
            // an upright image and all coordinates are in the same space as the preview.
            val bitmap = if (rotationDegrees != 0) {
                val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
                val rotated = Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true)
                rawBitmap.recycle()
                rotated
            } else {
                rawBitmap
            }

            // After rotation these dimensions match what the preview actually shows.
            previewFrameWidth = bitmap.width
            previewFrameHeight = bitmap.height
            try {
                val (displayBox, displayConfidence) = if (isDigitOnlyMode) {
                    buildCentralDisplayBox(bitmap.width, bitmap.height) to DIGIT_ONLY_SYNTHETIC_CONFIDENCE
                } else {
                    val displayModel = displayDetector ?: run {
                        isProcessing.set(false)
                        return
                    }
                    val (displayInputW, displayInputH) = displayModel.resolveInputSize(
                        defaultWidth = bitmap.width,
                        defaultHeight = bitmap.height
                    )
                    val displayInput = BitmapUtils.letterbox(
                        source = bitmap,
                        targetWidth = displayInputW,
                        targetHeight = displayInputH
                    )
                    try {
                        val displayDetections = displayModel.detect(displayInput.bitmap)
                            .map { remapFromLetterbox(it, displayInput, bitmap.width, bitmap.height) }
                            .sortedByDescending { it.conf }

                        val rawDisplay = displayDetections.firstOrNull()
                        Log.d(TAG, "analyzeImage: display=${rawDisplay?.let { "conf=${it.conf} box=${it.box}" } ?: "none"}")
                        val adjusted = rawDisplay?.let { det ->
                            val box = enforceDisplayAspectRatio(det.box, bitmap.width, bitmap.height)
                            det.copy(box = box, xCenter = box.centerX(), yCenter = box.centerY())
                        }
                        adjusted?.box to (adjusted?.conf ?: 0f)
                    } finally {
                        displayInput.bitmap.recycle()
                    }
                }

                if (displayBox != null && !isPhotoMode) {
                    lastLiveDisplayBox = RectF(displayBox)
                    lastLiveDisplayConfidence = displayConfidence
                    lastLiveSeenAtMs = now
                }

                when {
                    displayBox == null -> {
                        lastStableBox = null
                        stableSinceMs = 0L
                        val canHoldOverlay = !isPhotoMode &&
                            shouldHoldLiveOverlay(now) &&
                            lastLiveDisplayBox != null
                        runOnUiThread {
                            if (canHoldOverlay) {
                                val cachedDisplay = RectF(lastLiveDisplayBox!!)
                                binding.overlayView.update(
                                    cachedDisplay,
                                    lastLiveDigits,
                                    previewFrameWidth,
                                    previewFrameHeight
                                )
                                updateUi(
                                    state = ReaderState.PATROL,
                                    reading = lastLiveReading,
                                    message = "Señal inestable. Manteniendo última lectura (${formatPercent(lastLiveDisplayConfidence)} display${lastLiveReadingConfidence?.let { ", ${formatPercent(it)} dígitos" } ?: ""})."
                                )
                            } else {
                                binding.overlayView.update(null, emptyList(), previewFrameWidth, previewFrameHeight)
                                updateUi(
                                    state = ReaderState.PATROL,
                                    reading = null,
                                    message = if (isPhotoMode) {
                                        "No se detectó display. Presiona Capturar de nuevo."
                                    } else {
                                        "Buscando display..."
                                    }
                                )
                            }
                            if (isPhotoMode) showCaptureButton(true)
                        }
                        if (!canHoldOverlay) {
                            clearLiveCache()
                        }
                        isProcessing.set(false)
                    }

                    isPhotoMode -> {
                        runOnUiThread {
                            updateUi(
                                state = ReaderState.PROCESSING,
                                reading = null,
                                message = "Display detectado. Reconociendo dígitos..."
                            )
                            binding.overlayView.update(displayBox, emptyList(), previewFrameWidth, previewFrameHeight)
                        }
                        processFinalReading(bitmap, displayBox, liveMode = false)
                        isProcessing.set(false)
                    }

                    now - lastLiveDigitsInferenceAt >= liveDigitsIntervalMs -> {
                        runOnUiThread {
                            updateUi(
                                state = ReaderState.PROCESSING,
                                reading = lastLiveReading,
                                message = "Reconociendo en vivo... (display ${formatPercent(displayConfidence)}${lastLiveReadingConfidence?.let { ", dígitos ${formatPercent(it)}" } ?: ""})"
                            )
                            binding.overlayView.update(displayBox, lastLiveDigits, previewFrameWidth, previewFrameHeight)
                        }
                        processFinalReading(bitmap, displayBox, liveMode = true)
                        lastLiveDigitsInferenceAt = now
                        isProcessing.set(false)
                    }

                    else -> {
                        runOnUiThread {
                            binding.overlayView.update(displayBox, lastLiveDigits, previewFrameWidth, previewFrameHeight)
                            updateUi(
                                state = ReaderState.LOCKING,
                                reading = lastLiveReading,
                                message = if (isDigitOnlyMode) {
                                    "ROI central activo. Ajusta ángulo/distancia..."
                                } else {
                                    "Display ${formatPercent(displayConfidence)}${lastLiveReadingConfidence?.let { ", dígitos ${formatPercent(it)}" } ?: ""}."
                                }
                            )
                        }
                        isProcessing.set(false)
                    }
                }
            } finally {
                if (!bitmap.isRecycled) bitmap.recycle()
            }
        } catch (e: Exception) {
            Log.e(TAG, "analyzeImage error", e)
            imageProxy.closeSafely()
            isProcessing.set(false)
            runOnUiThread {
                updateUi(
                    state = if (isPhotoMode) ReaderState.ERROR else ReaderState.PATROL,
                    reading = null,
                    message = "Error durante el análisis: ${e.message}"
                )
            }
        }
    }

    private fun processFinalReading(frameBitmap: Bitmap, displayBox: RectF, liveMode: Boolean) {
        try {
            currentState = if (liveMode) ReaderState.PATROL else ReaderState.PROCESSING
            val digitModel = digitDetector ?: return

            val cropBounds = clampRectToBitmap(displayBox, frameBitmap.width, frameBitmap.height)
            Log.d(TAG, "processFinalReading: crop=$cropBounds (${cropBounds.width()}×${cropBounds.height()})")

            val crop = Bitmap.createBitmap(
                frameBitmap,
                cropBounds.left,
                cropBounds.top,
                cropBounds.width().coerceAtLeast(1),
                cropBounds.height().coerceAtLeast(1)
            )

            if (crop.width <= 1 || crop.height <= 1) {
                if (liveMode) {
                    runOnUiThread {
                        binding.overlayView.update(displayBox, emptyList(), previewFrameWidth, previewFrameHeight)
                        updateUi(
                            state = ReaderState.PATROL,
                            reading = "ilegible",
                            message = "Lectura en vivo: ilegible"
                        )
                    }
                } else {
                    showUnreadable(displayBox)
                    savePhotoPair(frameBitmap, displayBox, emptyList())
                }
                return
            }

            val candidate = try {
                buildBestCandidate(
                    crop = crop,
                    cropBounds = cropBounds,
                    digitModel = digitModel
                )
            } finally {
                if (!crop.isRecycled) crop.recycle()
            }

            Log.d(TAG, "processFinalReading: candidate=${candidate?.reading} digits=${candidate?.count}")

            val reading = candidate?.reading?.takeIf { it.isNotBlank() } ?: "ilegible"
            val overlayDigits = candidate?.fullFrameDetections ?: emptyList()
            val digitConfidence = candidate?.avgConfidence

            finalReading = reading
            if (liveMode) {
                currentState = ReaderState.PATROL
                lastLiveDigits = overlayDigits
                lastLiveReading = reading.takeIf { it != "ilegible" }
                lastLiveReadingConfidence = digitConfidence
                lastLiveSeenAtMs = SystemClock.elapsedRealtime()
                runOnUiThread {
                    binding.overlayView.update(displayBox, overlayDigits, previewFrameWidth, previewFrameHeight)
                    updateUi(
                        state = ReaderState.PATROL,
                        reading = reading,
                        message = if (reading == "ilegible") {
                            "Lectura en vivo: ilegible${digitConfidence?.let { " (${formatPercent(it)} confianza)" } ?: ""}"
                        } else {
                            "Lectura en vivo: $reading${digitConfidence?.let { " (${formatPercent(it)} confianza)" } ?: ""}"
                        }
                    )
                    showDecisionButtons(false)
                }
            } else {
                currentState = ReaderState.READY
                val capturesSaved = savePhotoPair(frameBitmap, displayBox, overlayDigits)
                runOnUiThread {
                    binding.overlayView.update(displayBox, overlayDigits, previewFrameWidth, previewFrameHeight)
                    updateUi(
                        state = ReaderState.READY,
                        reading = reading,
                        message = if (reading == "ilegible") {
                            "No se pudo construir una lectura confiable. Reintenta.${if (capturesSaved) " Capturas guardadas." else ""}"
                        } else {
                            "Lectura lista. Acepta o reintenta.${if (capturesSaved) " Capturas guardadas." else ""}"
                        }
                    )
                    showDecisionButtons(true)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "processFinalReading error", e)
            currentState = if (liveMode) ReaderState.PATROL else ReaderState.ERROR
            runOnUiThread {
                updateUi(
                    state = if (liveMode) ReaderState.PATROL else ReaderState.ERROR,
                    reading = null,
                    message = if (liveMode) "Lectura en vivo fallida, continuando..." else "Falló la inferencia final de dígitos."
                )
                if (!liveMode) showDecisionButtons(true)
            }
        }
    }

    /**
     * Letterbox the crop to the digit model expected input size, run the model,
     * and remap detected boxes back to full-frame coordinates.
     */
    private fun buildBestCandidate(
        crop: Bitmap,
        cropBounds: Rect,
        digitModel: OnnxYoloDetector
    ): CandidateResult? {
        Log.d(TAG, "buildBestCandidate: crop ${crop.width}×${crop.height}, running digit model")
        val (digitInputW, digitInputH) = digitModel.resolveInputSize(
            defaultWidth = crop.width,
            defaultHeight = crop.height
        )
        val lb = BitmapUtils.letterbox(
            source = crop,
            targetWidth = digitInputW,
            targetHeight = digitInputH
        )
        return try {
            evaluateVariant(lb.bitmap, digitModel) { box ->
                // Invert the letterbox transform to map model space back to crop space,
                // then shift by the crop origin to get full-frame coordinates.
                fun unmap(v: Float, offset: Float, max: Float) = ((v - offset) / lb.scale).coerceIn(0f, max)
                RectF(
                    unmap(box.left,   lb.dx, crop.width.toFloat())  + cropBounds.left,
                    unmap(box.top,    lb.dy, crop.height.toFloat()) + cropBounds.top,
                    unmap(box.right,  lb.dx, crop.width.toFloat())  + cropBounds.left,
                    unmap(box.bottom, lb.dy, crop.height.toFloat()) + cropBounds.top
                )
            }
        } finally {
            if (!lb.bitmap.isRecycled) lb.bitmap.recycle()
        }
    }

    private fun evaluateVariant(
        processedBitmap: Bitmap,
        digitModel: OnnxYoloDetector,
        mapper: (RectF) -> RectF
    ): CandidateResult? {
        val detections = digitModel.detect(processedBitmap)
        Log.d(TAG, "evaluateVariant: digitModel returned ${detections.size} raw detections")
        if (detections.isEmpty()) return null

        val rawDigits = detections.filterNot { it.isDot }
        val rawDots = detections.filter { it.isDot }
        val cleanDigits = ReadingLogic.solveOverlappingDigits(rawDigits, minDistPx = 20f)
        val allItems = (cleanDigits + rawDots).sortedBy { it.xCenter }
        val finalItems = ReadingLogic.filterDotsLogic(allItems).sortedBy { it.xCenter }
        if (finalItems.isEmpty()) return null

        val reading = ReadingLogic.buildReading(finalItems)
        val avgConf = finalItems.map { it.conf }.average().toFloat()
        val mapped = finalItems.map { det ->
            val fullBox = mapper(det.box)
            det.copy(
                xCenter = fullBox.centerX(),
                yCenter = fullBox.centerY(),
                box = fullBox
            )
        }.sortedBy { it.xCenter }

        return CandidateResult(
            reading = reading,
            avgConfidence = avgConf,
            count = finalItems.size,
            fullFrameDetections = mapped
        )
    }

    private fun showUnreadable(displayBox: RectF) {
        finalReading = "ilegible"
        currentState = ReaderState.READY
        runOnUiThread {
            binding.overlayView.update(displayBox, emptyList(), previewFrameWidth, previewFrameHeight)
            updateUi(
                state = ReaderState.READY,
                reading = "ilegible",
                message = "No se pudo construir una lectura confiable. Reintenta."
            )
            showDecisionButtons(true)
        }
    }

    private fun isStable(currentBox: RectF, now: Long): Boolean {
        val previous = lastStableBox
        if (previous == null) {
            lastStableBox = RectF(currentBox)
            stableSinceMs = now
            return false
        }

        val centerDx = abs(previous.centerX() - currentBox.centerX())
        val centerDy = abs(previous.centerY() - currentBox.centerY())
        val widthDiff = abs(previous.width() - currentBox.width())
        val heightDiff = abs(previous.height() - currentBox.height())

        val tolerant = centerDx < 18f && centerDy < 18f && widthDiff < 24f && heightDiff < 24f
        if (!tolerant) {
            lastStableBox = RectF(currentBox)
            stableSinceMs = now
            return false
        }

        lastStableBox = RectF(currentBox)
        return (now - stableSinceMs) >= stabilizationThresholdMs
    }

    private fun clampRectToBitmap(source: RectF, width: Int, height: Int): Rect {
        val left = source.left.toInt().coerceIn(0, (width - 1).coerceAtLeast(0))
        val top = source.top.toInt().coerceIn(0, (height - 1).coerceAtLeast(0))
        val right = source.right.toInt().coerceIn(left + 1, width.coerceAtLeast(1))
        val bottom = source.bottom.toInt().coerceIn(top + 1, height.coerceAtLeast(1))
        return Rect(left, top, right, bottom)
    }

    private fun remapFromLetterbox(
        det: Detection,
        letterbox: LetterboxResult,
        originalW: Int,
        originalH: Int
    ): Detection {
        val left = ((det.box.left - letterbox.dx) / letterbox.scale).coerceIn(0f, originalW.toFloat())
        val top = ((det.box.top - letterbox.dy) / letterbox.scale).coerceIn(0f, originalH.toFloat())
        val right = ((det.box.right - letterbox.dx) / letterbox.scale).coerceIn(0f, originalW.toFloat())
        val bottom = ((det.box.bottom - letterbox.dy) / letterbox.scale).coerceIn(0f, originalH.toFloat())
        val box = RectF(left, top, right, bottom)
        return det.copy(
            xCenter = box.centerX(),
            yCenter = box.centerY(),
            box = box
        )
    }

    /**
     * Adjust a detected display box so it always satisfies the 3.5:1 (width:height) aspect
     * ratio that is typical for electricity/gas meter display panels.  The box is expanded
     * on the shorter axis around the same centre and then clamped to the frame boundaries.
     */
    private fun enforceDisplayAspectRatio(box: RectF, frameW: Int, frameH: Int): RectF {
        val boxW = box.width()
        val boxH = box.height()
        if (boxW <= 0f || boxH <= 0f) return box

        val cx = box.centerX()
        val cy = box.centerY()
        val currentAspect = boxW / boxH

        val (newW, newH) = if (currentAspect < DISPLAY_ASPECT_RATIO) {
            // Too tall → widen to match target aspect ratio
            boxH * DISPLAY_ASPECT_RATIO to boxH
        } else {
            // Too wide → heighten to match target aspect ratio
            boxW to boxW / DISPLAY_ASPECT_RATIO
        }

        return RectF(
            (cx - newW / 2f).coerceAtLeast(0f),
            (cy - newH / 2f).coerceAtLeast(0f),
            (cx + newW / 2f).coerceAtMost(frameW.toFloat()),
            (cy + newH / 2f).coerceAtMost(frameH.toFloat())
        )
    }

    private fun buildCentralDisplayBox(frameW: Int, frameH: Int): RectF {
        val width = frameW * CENTRAL_ROI_WIDTH_RATIO
        val height = width / DISPLAY_ASPECT_RATIO
        val cx = frameW / 2f
        val cy = frameH / 2f
        return RectF(
            cx - width / 2f,
            cy - height / 2f,
            cx + width / 2f,
            cy + height / 2f
        )
    }

    private fun savePhotoPair(frameBitmap: Bitmap, displayBox: RectF, digits: List<Detection>): Boolean {
        var scaledRaw: Bitmap? = null
        var createdScaled = false
        var annotated: Bitmap? = null
        return try {
            val directory = File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "captures").apply { mkdirs() }
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
            val rawFile = File(directory, "${timestamp}_sin_reconocimiento.jpg")
            val recognizedFile = File(directory, "${timestamp}_con_reconocimiento.jpg")
            val scaledInfo = scaleDownForCapture(frameBitmap, maxEdge = MAX_CAPTURE_EDGE)
            scaledRaw = scaledInfo.first
            createdScaled = scaledInfo.second

            FileOutputStream(rawFile).use { output ->
                scaledRaw.compress(Bitmap.CompressFormat.JPEG, 90, output)
            }

            val sx = scaledRaw.width / frameBitmap.width.toFloat()
            val sy = scaledRaw.height / frameBitmap.height.toFloat()
            val scaledDisplay = RectF(displayBox.left * sx, displayBox.top * sy, displayBox.right * sx, displayBox.bottom * sy)
            val scaledDigits = digits.map { det ->
                val box = RectF(det.box.left * sx, det.box.top * sy, det.box.right * sx, det.box.bottom * sy)
                det.copy(box = box, xCenter = box.centerX(), yCenter = box.centerY())
            }

            annotated = drawRecognizedBitmap(scaledRaw, scaledDisplay, scaledDigits)
            FileOutputStream(recognizedFile).use { output ->
                annotated.compress(Bitmap.CompressFormat.JPEG, 90, output)
            }
            cleanupOldCaptures(directory)
            true
        } catch (e: Exception) {
            Log.e(TAG, "savePhotoPair error", e)
            false
        } finally {
            annotated?.let { if (it !== scaledRaw && !it.isRecycled) it.recycle() }
            scaledRaw?.let { if (createdScaled && !it.isRecycled) it.recycle() }
        }
    }

    private fun scaleDownForCapture(source: Bitmap, maxEdge: Int): Pair<Bitmap, Boolean> {
        val maxCurrent = maxOf(source.width, source.height)
        if (maxCurrent <= maxEdge) return source to false
        val scale = maxEdge / maxCurrent.toFloat()
        val targetW = (source.width * scale).toInt().coerceAtLeast(1)
        val targetH = (source.height * scale).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(source, targetW, targetH, true) to true
    }

    private fun cleanupOldCaptures(directory: File) {
        val files = directory.listFiles()?.filter { it.isFile } ?: return
        if (files.size <= MAX_CAPTURE_FILES) return
        files.sortedBy { it.lastModified() }
            .take(files.size - MAX_CAPTURE_FILES)
            .forEach {
                if (!it.delete()) {
                    Log.w(TAG, "No se pudo eliminar captura antigua: ${it.name}")
                }
            }
    }

    private fun drawRecognizedBitmap(source: Bitmap, displayBox: RectF, digits: List<Detection>): Bitmap {
        val output = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)

        val displayPaint = Paint().apply {
            color = Color.YELLOW
            style = Paint.Style.STROKE
            strokeWidth = 6f
        }
        val digitPaint = Paint().apply {
            color = Color.CYAN
            style = Paint.Style.STROKE
            strokeWidth = 4f
        }
        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 32f
            style = Paint.Style.FILL
            isAntiAlias = true
        }
        val textBgPaint = Paint().apply {
            color = Color.argb(180, 0, 0, 0)
            style = Paint.Style.FILL
        }

        canvas.drawRect(displayBox, displayPaint)
        for (det in digits) {
            canvas.drawRect(det.box, digitPaint)
            val label = "${det.label} ${(det.conf * 100).toInt()}%"
            val textW = textPaint.measureText(label)
            val textH = textPaint.textSize + 14f
            val top = (det.box.top - textH).coerceAtLeast(0f)
            canvas.drawRect(det.box.left, top, det.box.left + textW + 16f, top + textH, textBgPaint)
            canvas.drawText(label, det.box.left + 8f, top + textH - 10f, textPaint)
        }
        return output
    }

    private fun ImageProxy.toBitmap(): Bitmap {
        val plane = planes[0]
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val rowPadding = rowStride - pixelStride * width
        val paddedWidth = width + rowPadding / pixelStride

        val bitmap = Bitmap.createBitmap(paddedWidth, height, Bitmap.Config.ARGB_8888)
        buffer.rewind()
        bitmap.copyPixelsFromBuffer(buffer)
        return Bitmap.createBitmap(bitmap, 0, 0, width, height)
    }

    private fun ImageProxy.closeSafely() {
        try {
            close()
        } catch (_: Exception) {
        }
    }

    private fun setupButtons() {
        binding.btnRetry.setOnClickListener {
            resetToPatrol()
        }
        binding.btnAccept.setOnClickListener {
            updateUi(
                state = ReaderState.READY,
                reading = finalReading,
                message = "Lectura aceptada. Pulsa Reintentar para capturar otra."
            )
        }
        binding.btnToggleMode.setOnClickListener {
            toggleMode()
        }
        binding.btnToggleDetector.setOnClickListener {
            isDigitOnlyMode = !isDigitOnlyMode
            updateDetectorModeButton()
            resetToPatrol()
            updateUi(
                state = ReaderState.PATROL,
                reading = null,
                message = if (isDigitOnlyMode) {
                    "Modo rápido: reconocimiento de números en ROI central."
                } else {
                    "Modo estándar: detección de display + números."
                }
            )
        }
        binding.btnCapture.setOnClickListener {
            captureNextFrame.set(true)
            showCaptureButton(false)
            updateUi(
                state = ReaderState.PATROL,
                reading = null,
                message = "Procesando captura..."
            )
        }
        updateDetectorModeButton()
    }

    private fun updateDetectorModeButton() {
        binding.btnToggleDetector.text = if (isDigitOnlyMode) {
            getString(R.string.detector_digits_only)
        } else {
            getString(R.string.detector_display)
        }
    }

    private fun toggleMode() {
        isPhotoMode = !isPhotoMode
        captureNextFrame.set(false)
        finalReading = null
        lastStableBox = null
        stableSinceMs = 0L
        clearLiveCache()
        currentState = ReaderState.PATROL
        showDecisionButtons(false)
        binding.overlayView.clearAll()

        if (isPhotoMode) {
            binding.btnToggleMode.text = getString(R.string.mode_photo)
            showCaptureButton(true)
            updateUi(
                state = ReaderState.PATROL,
                reading = null,
                message = "Modo Foto: presiona Capturar para fotografiar el display."
            )
        } else {
            binding.btnToggleMode.text = getString(R.string.mode_live)
            showCaptureButton(false)
            updateUi(
                state = ReaderState.PATROL,
                reading = null,
                message = "Modo Vivo: apunta al display. Lectura continua activada."
            )
        }
    }

    private fun resetToPatrol() {
        finalReading = null
        lastStableBox = null
        stableSinceMs = 0L
        clearLiveCache()
        currentState = ReaderState.PATROL
        captureNextFrame.set(false)
        showDecisionButtons(false)
        binding.overlayView.clearAll()

        if (isPhotoMode) {
            showCaptureButton(true)
            updateUi(
                state = ReaderState.PATROL,
                reading = null,
                message = "Modo Foto: presiona Capturar para fotografiar el display."
            )
        } else {
                updateUi(
                    state = ReaderState.PATROL,
                    reading = null,
                    message = "Apunta al display. Lectura continua activada."
                )
            }
        }

    private fun showDecisionButtons(show: Boolean) {
        binding.btnAccept.visibility = if (show) View.VISIBLE else View.GONE
        binding.btnRetry.visibility = if (show) View.VISIBLE else View.GONE
        // While showing the result, hide Capture so the UI isn't cluttered.
        if (show) showCaptureButton(false)
    }

    private fun showCaptureButton(show: Boolean) {
        binding.btnCapture.visibility = if (show) View.VISIBLE else View.GONE
    }

    private fun updateUi(state: ReaderState, reading: String?, message: String) {
        currentState = state
        binding.tvState.text = when (state) {
            ReaderState.PATROL -> getString(R.string.state_patrol)
            ReaderState.LOCKING -> getString(R.string.state_locking)
            ReaderState.PROCESSING -> getString(R.string.state_processing)
            ReaderState.READY -> getString(R.string.state_ready)
            ReaderState.ERROR -> getString(R.string.state_error)
        }
        binding.tvResult.text = "Lectura: ${reading ?: "--"}"
        binding.tvHint.text = message
    }

    private fun formatPercent(value: Float): String {
        return "${(value * 100f).toInt().coerceIn(0, 100)}%"
    }

    private fun shouldHoldLiveOverlay(now: Long): Boolean {
        return (now - lastLiveSeenAtMs) <= LIVE_OVERLAY_HOLD_MS
    }

    private fun clearLiveCache() {
        lastLiveDisplayBox = null
        lastLiveDisplayConfidence = 0f
        lastLiveDigits = emptyList()
        lastLiveReading = null
        lastLiveReadingConfidence = null
        lastLiveSeenAtMs = 0L
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        displayDetector?.close()
        digitDetector?.close()
    }

    private data class CandidateResult(
        val reading: String,
        val avgConfidence: Float,
        val count: Int,
        val fullFrameDetections: List<Detection>
    )
}
