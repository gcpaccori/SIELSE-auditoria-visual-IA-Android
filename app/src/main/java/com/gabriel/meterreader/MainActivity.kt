package com.gabriel.meterreader

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.RectF
import android.os.Bundle
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
import kotlin.math.abs

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService

    private var displayDetector: OnnxYoloDetector? = null
    private var digitDetector: OnnxYoloDetector? = null

    private val isProcessing = AtomicBoolean(false)

    private var currentState: ReaderState = ReaderState.PATROL
    private var lastPatrolInferenceAt = 0L
    private val patrolIntervalMs = 500L

    private var lastStableBox: RectF? = null
    private var stableSinceMs = 0L
    private val stabilizationThresholdMs = 800L

    private var previewFrameWidth: Int = 1
    private var previewFrameHeight: Int = 1
    private var finalReading: String? = null

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
                    inputSize = 320,
                    confidenceThreshold = 0.25f,
                    iouThreshold = 0.50f
                )
            )
            digitDetector = OnnxYoloDetector(
                context = applicationContext,
                config = ModelConfig(
                    modelFile = "digit_recognition_int8.onnx",
                    labelsFile = "labels_digits.txt",
                    inputSize = 320,
                    confidenceThreshold = 0.30f,
                    iouThreshold = 0.20f
                )
            )
            withContext(Dispatchers.Main) {
                updateUi(
                    state = ReaderState.PATROL,
                    reading = null,
                    message = "Apunta al medidor. El display se analiza a 2 FPS."
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
            if (currentState == ReaderState.READY || currentState == ReaderState.PROCESSING) {
                imageProxy.close()
                return
            }

            val now = SystemClock.elapsedRealtime()
            if (now - lastPatrolInferenceAt < patrolIntervalMs) {
                imageProxy.close()
                return
            }

            if (!isProcessing.compareAndSet(false, true)) {
                imageProxy.close()
                return
            }

            lastPatrolInferenceAt = now
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

            val displayModel = displayDetector ?: run {
                isProcessing.set(false)
                return
            }

            val displayInput = BitmapUtils.letterboxToSquare(bitmap, 320)
            val displayDetections = displayModel.detect(displayInput.bitmap)
                .map { remapFromLetterbox(it, displayInput, bitmap.width, bitmap.height) }
                .sortedByDescending { it.conf }

            val display = displayDetections.firstOrNull()

            when {
                display == null -> {
                    lastStableBox = null
                    stableSinceMs = 0L
                    runOnUiThread {
                        binding.overlayView.update(null, emptyList(), previewFrameWidth, previewFrameHeight)
                        updateUi(
                            state = ReaderState.PATROL,
                            reading = null,
                            message = "Buscando display..."
                        )
                    }
                    isProcessing.set(false)
                }

                isStable(display.box, now) -> {
                    runOnUiThread {
                        updateUi(
                            state = ReaderState.PROCESSING,
                            reading = null,
                            message = "Display estable. Ejecutando reconocimiento de dígitos..."
                        )
                        binding.overlayView.update(display.box, emptyList(), previewFrameWidth, previewFrameHeight)
                    }
                    processFinalReading(bitmap, display.box)
                    isProcessing.set(false)
                }

                else -> {
                    runOnUiThread {
                        binding.overlayView.update(display.box, emptyList(), previewFrameWidth, previewFrameHeight)
                        updateUi(
                            state = ReaderState.LOCKING,
                            reading = null,
                            message = "Mantenga quieto el dispositivo..."
                        )
                    }
                    isProcessing.set(false)
                }
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "analyzeImage error", e)
            imageProxy.closeSafely()
            isProcessing.set(false)
            runOnUiThread {
                updateUi(
                    state = ReaderState.ERROR,
                    reading = null,
                    message = "Error durante el análisis: ${e.message}"
                )
            }
        }
    }

    private fun processFinalReading(frameBitmap: Bitmap, displayBox: RectF) {
        try {
            currentState = ReaderState.PROCESSING
            val digitModel = digitDetector ?: return

            val cropBounds = clampRectToBitmap(displayBox, frameBitmap.width, frameBitmap.height)
            val crop = Bitmap.createBitmap(
                frameBitmap,
                cropBounds.left,
                cropBounds.top,
                cropBounds.width().coerceAtLeast(1),
                cropBounds.height().coerceAtLeast(1)
            )

            if (crop.width <= 1 || crop.height <= 1) {
                showUnreadable(displayBox)
                return
            }

            val candidate = buildBestCandidate(
                crop = crop,
                cropBounds = cropBounds,
                digitModel = digitModel
            )

            val reading = candidate?.reading?.takeIf { it.isNotBlank() } ?: "ilegible"
            val overlayDigits = candidate?.fullFrameDetections ?: emptyList()

            finalReading = reading
            currentState = ReaderState.READY

            runOnUiThread {
                binding.overlayView.update(displayBox, overlayDigits, previewFrameWidth, previewFrameHeight)
                updateUi(
                    state = ReaderState.READY,
                    reading = reading,
                    message = if (reading == "ilegible") {
                        "No se pudo construir una lectura confiable. Reintenta."
                    } else {
                        "Lectura lista. Acepta o reintenta."
                    }
                )
                showDecisionButtons(true)
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "processFinalReading error", e)
            currentState = ReaderState.ERROR
            runOnUiThread {
                updateUi(
                    state = ReaderState.ERROR,
                    reading = null,
                    message = "Falló la inferencia final de dígitos."
                )
                showDecisionButtons(true)
            }
        }
    }

    private fun buildBestCandidate(
        crop: Bitmap,
        cropBounds: Rect,
        digitModel: OnnxYoloDetector
    ): CandidateResult? {
        val ratio = crop.width.toFloat() / crop.height.toFloat()

        return when {
            ratio < 0.85f -> {
                val leftRotated = BitmapUtils.rotate90Ccw(crop)
                val leftProcessed = BitmapUtils.resize(leftRotated, 400, 150)
                val leftResult = evaluateVariant(leftProcessed, digitModel) { box ->
                    mapProcessedVerticalLeftBoxToFullFrame(box, crop.width, crop.height, cropBounds)
                }

                val rightRotated = BitmapUtils.rotate90Cw(crop)
                val rightProcessed = BitmapUtils.resize(rightRotated, 400, 150)
                val rightResult = evaluateVariant(rightProcessed, digitModel) { box ->
                    mapProcessedVerticalRightBoxToFullFrame(box, crop.width, crop.height, cropBounds)
                }

                listOfNotNull(leftResult, rightResult)
                    .maxByOrNull { (it.count * 10f) + it.avgConfidence }
            }

            ratio <= 1.3f -> {
                val targetW = 350
                val scale = targetW / crop.width.toFloat()
                val targetH = (crop.height * scale).toInt().coerceAtLeast(1)
                val processed = BitmapUtils.resize(crop, targetW, targetH)
                evaluateVariant(processed, digitModel) { box ->
                    mapProcessedSquareBoxToFullFrame(box, scale, cropBounds)
                }
            }

            else -> {
                val processed = BitmapUtils.resize(crop, 400, 150)
                evaluateVariant(processed, digitModel) { box ->
                    mapProcessedHorizontalBoxToFullFrame(box, crop.width, crop.height, cropBounds)
                }
            }
        }
    }

    private fun evaluateVariant(
        processedBitmap: Bitmap,
        digitModel: OnnxYoloDetector,
        mapper: (RectF) -> RectF
    ): CandidateResult? {
        val detections = digitModel.detect(processedBitmap)
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

    private fun mapProcessedHorizontalBoxToFullFrame(
        box: RectF,
        cropW: Int,
        cropH: Int,
        cropBounds: Rect
    ): RectF {
        val sx = cropW / 400f
        val sy = cropH / 150f
        return offsetRect(
            RectF(box.left * sx, box.top * sy, box.right * sx, box.bottom * sy),
            cropBounds.left.toFloat(),
            cropBounds.top.toFloat()
        )
    }

    private fun mapProcessedSquareBoxToFullFrame(
        box: RectF,
        scale: Float,
        cropBounds: Rect
    ): RectF {
        return offsetRect(
            RectF(box.left / scale, box.top / scale, box.right / scale, box.bottom / scale),
            cropBounds.left.toFloat(),
            cropBounds.top.toFloat()
        )
    }

    private fun mapProcessedVerticalLeftBoxToFullFrame(
        box: RectF,
        originalW: Int,
        originalH: Int,
        cropBounds: Rect
    ): RectF {
        val rotW = originalH.toFloat()
        val rotH = originalW.toFloat()
        val sx = rotW / 400f
        val sy = rotH / 150f
        val inRotated = RectF(box.left * sx, box.top * sy, box.right * sx, box.bottom * sy)
        val inOriginal = mapRectByCorners(inRotated) { xR, yR ->
            val xO = originalW - yR
            val yO = xR
            xO to yO
        }
        return offsetRect(inOriginal, cropBounds.left.toFloat(), cropBounds.top.toFloat())
    }

    private fun mapProcessedVerticalRightBoxToFullFrame(
        box: RectF,
        originalW: Int,
        originalH: Int,
        cropBounds: Rect
    ): RectF {
        val rotW = originalH.toFloat()
        val rotH = originalW.toFloat()
        val sx = rotW / 400f
        val sy = rotH / 150f
        val inRotated = RectF(box.left * sx, box.top * sy, box.right * sx, box.bottom * sy)
        val inOriginal = mapRectByCorners(inRotated) { xR, yR ->
            val xO = yR
            val yO = originalH - xR
            xO to yO
        }
        return offsetRect(inOriginal, cropBounds.left.toFloat(), cropBounds.top.toFloat())
    }

    private fun offsetRect(rect: RectF, dx: Float, dy: Float): RectF {
        return RectF(rect.left + dx, rect.top + dy, rect.right + dx, rect.bottom + dy)
    }

    private fun mapRectByCorners(
        source: RectF,
        mapper: (Float, Float) -> Pair<Float, Float>
    ): RectF {
        val points = listOf(
            mapper(source.left, source.top),
            mapper(source.right, source.top),
            mapper(source.left, source.bottom),
            mapper(source.right, source.bottom)
        )
        val minX = points.minOf { it.first }
        val minY = points.minOf { it.second }
        val maxX = points.maxOf { it.first }
        val maxY = points.maxOf { it.second }
        return RectF(minX, minY, maxX, maxY)
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
    }

    private fun resetToPatrol() {
        finalReading = null
        lastStableBox = null
        stableSinceMs = 0L
        currentState = ReaderState.PATROL
        showDecisionButtons(false)
        binding.overlayView.clearAll()
        updateUi(
            state = ReaderState.PATROL,
            reading = null,
            message = "Apunta al display. El análisis corre a 2 FPS."
        )
    }

    private fun showDecisionButtons(show: Boolean) {
        binding.btnAccept.visibility = if (show) View.VISIBLE else View.GONE
        binding.btnRetry.visibility = if (show) View.VISIBLE else View.GONE
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
