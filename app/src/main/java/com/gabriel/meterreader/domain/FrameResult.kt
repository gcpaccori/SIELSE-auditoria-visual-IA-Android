package com.gabriel.meterreader.domain

import android.graphics.RectF

data class FrameResult(
    val state: ReaderState,
    val displayBox: RectF? = null,
    val digitBoxes: List<Detection> = emptyList(),
    val reading: String? = null,
    val message: String = ""
)

enum class ReaderState {
    PATROL,
    LOCKING,
    PROCESSING,
    READY,
    ERROR
}
