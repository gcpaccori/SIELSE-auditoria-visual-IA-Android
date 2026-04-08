package com.gabriel.meterreader.util

import android.content.Context
import java.io.File

object AssetUtils {
    fun copyAssetToCache(context: Context, fileName: String): File {
        val outFile = File(context.cacheDir, fileName)
        if (outFile.exists() && outFile.length() > 0L) return outFile

        context.assets.open(fileName).use { input ->
            outFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }
        return outFile
    }

    fun loadLabels(context: Context, fileName: String): List<String> {
        return context.assets.open(fileName).bufferedReader().useLines { lines ->
            lines.map { it.trim() }.filter { it.isNotEmpty() }.toList()
        }
    }
}
