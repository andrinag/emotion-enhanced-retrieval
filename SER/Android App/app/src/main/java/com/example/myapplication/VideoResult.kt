package com.example.myapplication

/**
 * Data Class containing the necessary info for query results.
 */
data class VideoResult(
    val videoUrl: String,
    val frameTime: Double,
    val annotatedImageUrl: String?
)
