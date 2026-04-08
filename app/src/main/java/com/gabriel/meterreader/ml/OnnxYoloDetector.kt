package com.gabriel.meterreader.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.gabriel.meterreader.domain.Detection
import com.gabriel.meterreader.util.AssetUtils
import kotlin.math.max
import kotlin.math.min

class OnnxYoloDetector(
    context: Context,
    private val config: ModelConfig
) : AutoCloseable {

    private val labels: List<String> = AssetUtils.loadLabels(context, config.labelsFile)
    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val modelPath: String = AssetUtils.copyAssetToCache(context, config.modelFile).absolutePath

    private val session: OrtSession by lazy {
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setInterOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        environment.createSession(modelPath, options)
    }

    private val inputName: String by lazy { session.inputNames.first() }

    fun detect(bitmap: Bitmap): List<Detection> {
        val inputTensor = bitmapToTensor(bitmap)
        try {
            val results = session.run(mapOf(inputName to inputTensor))
            try {
                val outputValue = results[0].value
                return parseDetections(outputValue, bitmap.width, bitmap.height)
            } finally {
                results.close()
            }
        } finally {
            inputTensor.close()
        }
    }

    private fun bitmapToTensor(bitmap: Bitmap): OnnxTensor {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val chw = FloatArray(1 * 3 * height * width)
        val hw = height * width

        for (y in 0 until height) {
            for (x in 0 until width) {
                val px = pixels[y * width + x]
                val r = ((px shr 16) and 0xFF) / 255f
                val g = ((px shr 8) and 0xFF) / 255f
                val b = (px and 0xFF) / 255f
                val idx = y * width + x
                chw[idx] = r
                chw[hw + idx] = g
                chw[(2 * hw) + idx] = b
            }
        }

        return OnnxTensor.createTensor(environment, java.nio.FloatBuffer.wrap(chw), longArrayOf(1, 3, height.toLong(), width.toLong()))
    }

    private fun parseDetections(
        outputValue: Any?,
        frameWidth: Int,
        frameHeight: Int
    ): List<Detection> {
        val raw3d = outputValue as? Array<*> ?: return emptyList()
        if (raw3d.isEmpty()) return emptyList()
        val batch0 = raw3d[0] as? Array<*> ?: return emptyList()
        if (batch0.isEmpty()) return emptyList()

        val candidates = mutableListOf<Detection>()

        val firstInner = batch0[0] as? FloatArray ?: return emptyList()
        val dimA = batch0.size
        val dimB = firstInner.size

        if (dimA >= 5 && dimB >= 1) {
            // Typical Ultralytics ONNX detect output: [1, channels, num_boxes]
            val channels = dimA
            val rows = dimB
            for (i in 0 until rows) {
                val row = FloatArray(channels)
                for (c in 0 until channels) {
                    row[c] = (batch0[c] as FloatArray)[i]
                }
                rowToDetection(row, frameWidth, frameHeight)?.let { candidates += it }
            }
        } else if (dimB >= 5) {
            // Alternate layout: [1, num_boxes, channels]
            for (rowAny in batch0) {
                val row = rowAny as? FloatArray ?: continue
                rowToDetection(row, frameWidth, frameHeight)?.let { candidates += it }
            }
        }

        return nms(candidates.filter { it.conf >= config.confidenceThreshold }, config.iouThreshold)
    }

    private fun rowToDetection(row: FloatArray, frameWidth: Int, frameHeight: Int): Detection? {
        val classStart = if (config.hasObjectness) 5 else 4
        if (row.size <= classStart) return null

        var cx = row[0]
        var cy = row[1]
        var w = row[2]
        var h = row[3]

        val normalized = cx <= 1.5f && cy <= 1.5f && w <= 1.5f && h <= 1.5f
        if (normalized) {
            cx *= frameWidth
            cy *= frameHeight
            w *= frameWidth
            h *= frameHeight
        }

        val classSlice = row.copyOfRange(classStart, row.size)
        if (classSlice.isEmpty()) return null

        val best = classSlice.withIndex().maxByOrNull { it.value } ?: return null
        val classIndex = best.index
        val classScore = best.value
        val confidence = if (config.hasObjectness) row[4] * classScore else classScore
        if (confidence < config.confidenceThreshold) return null

        val left = (cx - w / 2f).coerceIn(0f, frameWidth.toFloat())
        val top = (cy - h / 2f).coerceIn(0f, frameHeight.toFloat())
        val right = (cx + w / 2f).coerceIn(0f, frameWidth.toFloat())
        val bottom = (cy + h / 2f).coerceIn(0f, frameHeight.toFloat())
        if (right <= left || bottom <= top) return null

        val label = labels.getOrNull(classIndex) ?: classIndex.toString()
        val normalizedLabel = label.lowercase()
        val isDot = normalizedLabel == "." || normalizedLabel == "10" || normalizedLabel.contains("dot") || normalizedLabel.contains("point")

        return Detection(
            xCenter = (left + right) / 2f,
            yCenter = (top + bottom) / 2f,
            label = label,
            conf = confidence,
            isDot = isDot,
            box = RectF(left, top, right, bottom),
            classIndex = classIndex
        )
    }

    private fun nms(items: List<Detection>, iouThreshold: Float): List<Detection> {
        val sorted = items.sortedByDescending { it.conf }.toMutableList()
        val selected = mutableListOf<Detection>()

        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            selected += best
            val iterator = sorted.iterator()
            while (iterator.hasNext()) {
                val other = iterator.next()
                if (iou(best.box, other.box) > iouThreshold) {
                    iterator.remove()
                }
            }
        }
        return selected
    }

    private fun iou(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        val interW = max(0f, right - left)
        val interH = max(0f, bottom - top)
        val interArea = interW * interH
        val union = a.width() * a.height() + b.width() * b.height() - interArea
        return if (union <= 0f) 0f else interArea / union
    }

    override fun close() {
        session.close()
    }
}
