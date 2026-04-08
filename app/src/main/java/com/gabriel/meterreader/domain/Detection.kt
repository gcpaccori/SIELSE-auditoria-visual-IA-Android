package com.gabriel.meterreader.domain

import android.graphics.RectF

data class Detection(
    val xCenter: Float,
    val yCenter: Float,
    val label: String,
    val conf: Float,
    val isDot: Boolean,
    val box: RectF,
    val classIndex: Int
)
