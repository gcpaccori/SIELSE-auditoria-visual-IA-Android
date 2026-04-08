package com.gabriel.meterreader.util

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF

data class LetterboxResult(
    val bitmap: Bitmap,
    val scale: Float,
    val dx: Float,
    val dy: Float
)

object BitmapUtils {
    fun rgba8888ToBitmap(width: Int, height: Int, bytes: ByteArray): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(java.nio.ByteBuffer.wrap(bytes))
        return bitmap
    }

    fun cropBitmapSafe(source: Bitmap, rect: RectF): Bitmap {
        val left = rect.left.coerceAtLeast(0f).toInt()
        val top = rect.top.coerceAtLeast(0f).toInt()
        val right = rect.right.coerceAtMost(source.width.toFloat()).toInt()
        val bottom = rect.bottom.coerceAtMost(source.height.toFloat()).toInt()
        val width = (right - left).coerceAtLeast(1)
        val height = (bottom - top).coerceAtLeast(1)
        return Bitmap.createBitmap(source, left, top, width, height)
    }

    fun resize(source: Bitmap, width: Int, height: Int): Bitmap {
        return Bitmap.createScaledBitmap(source, width.coerceAtLeast(1), height.coerceAtLeast(1), true)
    }

    fun rotate90Ccw(source: Bitmap): Bitmap {
        val matrix = Matrix().apply { postRotate(-90f) }
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
    }

    fun rotate90Cw(source: Bitmap): Bitmap {
        val matrix = Matrix().apply { postRotate(90f) }
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
    }

    fun letterboxToSquare(source: Bitmap, size: Int): LetterboxResult {
        val srcW = source.width.toFloat()
        val srcH = source.height.toFloat()
        val scale = minOf(size / srcW, size / srcH)
        val dstW = (srcW * scale).toInt().coerceAtLeast(1)
        val dstH = (srcH * scale).toInt().coerceAtLeast(1)
        val dx = ((size - dstW) / 2f)
        val dy = ((size - dstH) / 2f)

        val output = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(output)
        canvas.drawColor(Color.DKGRAY)
        val dst = RectF(dx, dy, dx + dstW, dy + dstH)
        canvas.drawBitmap(source, null, dst, Paint(Paint.FILTER_BITMAP_FLAG))
        return LetterboxResult(output, scale, dx, dy)
    }
}
