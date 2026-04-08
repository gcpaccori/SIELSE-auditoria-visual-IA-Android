package com.gabriel.meterreader.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.gabriel.meterreader.domain.Detection

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val displayPaint = Paint().apply {
        color = Color.YELLOW
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    private val digitPaint = Paint().apply {
        color = Color.CYAN
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 34f
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val textBg = Paint().apply {
        color = Color.argb(150, 0, 0, 0)
        style = Paint.Style.FILL
    }

    private var displayBox: RectF? = null
    private var digits: List<Detection> = emptyList()
    private var previewWidth: Int = 1
    private var previewHeight: Int = 1

    fun update(
        displayBox: RectF?,
        digits: List<Detection>,
        previewWidth: Int,
        previewHeight: Int
    ) {
        this.displayBox = displayBox
        this.digits = digits
        this.previewWidth = previewWidth.coerceAtLeast(1)
        this.previewHeight = previewHeight.coerceAtLeast(1)
        invalidate()
    }

    fun clearAll() {
        displayBox = null
        digits = emptyList()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        drawBox(canvas, displayBox, displayPaint, "DISPLAY")
        digits.forEach { det ->
            drawBox(canvas, det.box, digitPaint, "${det.label} ${(det.conf * 100).toInt()}%")
        }
    }

    private fun drawBox(canvas: Canvas, source: RectF?, paint: Paint, label: String) {
        if (source == null) return
        val mapped = mapRect(source)
        canvas.drawRect(mapped, paint)

        val textWidth = textPaint.measureText(label)
        val textHeight = textPaint.textSize + 16f
        val top = (mapped.top - textHeight).coerceAtLeast(0f)
        canvas.drawRect(mapped.left, top, mapped.left + textWidth + 20f, top + textHeight, textBg)
        canvas.drawText(label, mapped.left + 10f, top + textHeight - 12f, textPaint)
    }

    private fun mapRect(source: RectF): RectF {
        val scaleX = width / previewWidth.toFloat()
        val scaleY = height / previewHeight.toFloat()
        return RectF(
            source.left * scaleX,
            source.top * scaleY,
            source.right * scaleX,
            source.bottom * scaleY
        )
    }
}
