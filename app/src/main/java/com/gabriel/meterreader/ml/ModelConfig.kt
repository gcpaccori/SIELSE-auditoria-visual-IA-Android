package com.gabriel.meterreader.ml

data class ModelConfig(
    val modelFile: String,
    val labelsFile: String,
    val inputSize: Int,
    val confidenceThreshold: Float,
    val iouThreshold: Float,
    val hasObjectness: Boolean = false
)
