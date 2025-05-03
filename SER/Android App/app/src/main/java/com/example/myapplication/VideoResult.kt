package com.example.myapplication

/**
 * Data Class containing the necessary info for query results.
 */
data class VideoResult(
    val videoUrl: String,
    val frameTime: Double,
    val annotatedImageUrl: String? = null,
    val frameLocation: String? = null,
    val embeddingID: Int,
    val previousEmbeddingID: Int
)
