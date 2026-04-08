package com.gabriel.meterreader.ml

import com.gabriel.meterreader.domain.Detection
import kotlin.math.abs

object ReadingLogic {

    fun solveOverlappingDigits(
        detections: List<Detection>,
        minDistPx: Float = 20f
    ): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        val sorted = detections.sortedByDescending { it.conf }
        val accepted = mutableListOf<Detection>()

        for (candidate in sorted) {
            val collides = accepted.any { existing ->
                abs(existing.xCenter - candidate.xCenter) < minDistPx
            }
            if (!collides) {
                accepted += candidate
            }
        }

        return accepted.sortedBy { it.xCenter }
    }

    fun filterDotsLogic(detections: List<Detection>): List<Detection> {
        val sorted = detections.sortedBy { it.xCenter }
        val dots = sorted.filter { it.isDot }
        if (dots.size <= 1) return sorted

        val lastDot = dots.maxByOrNull { it.xCenter }
        return sorted.filter { !it.isDot } + listOfNotNull(lastDot)
    }

    fun buildReading(detections: List<Detection>): String {
        return detections
            .sortedBy { it.xCenter }
            .joinToString(separator = "") { det ->
                if (det.isDot) "." else det.label.filter { ch -> ch.isDigit() }
            }
            .replace("..", ".")
    }
}
