package com.example.myapplication

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import androidx.recyclerview.widget.RecyclerView
import com.bumptech.glide.Glide

/**
 * RecyclerView Adapter to show only thumbnails.
 * When clicked, opens full video in a separate activity.
 */
class ResultsAdapter(
    private val items: List<VideoResult>,
    private val context: Context,
    private val query: String,
    private val emotion: String,
    private val dataType: String
) : RecyclerView.Adapter<ResultsAdapter.ViewHolder>() {

    inner class ViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val imageView: ImageView = itemView.findViewById(R.id.annotationImage)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.result_item, parent, false)
        return ViewHolder(view)
    }

    /**
     * Generates a Video Thumbnail for the videos that have no annotated image or frame location.
     * Should technically never be called, but is left in code for emergencies.
     */
    fun generateVideoThumbnail(videoUrl: String, frameTimeMillis: Long): Bitmap? {
        return try {
            val retriever = MediaMetadataRetriever()
            retriever.setDataSource(videoUrl, HashMap()) // Use empty headers map
            val bitmap = retriever.getFrameAtTime(frameTimeMillis * 1000, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
            retriever.release()
            Log.d("Thumbnail", "Generated thumbnail at $frameTimeMillis ms")
            bitmap
        } catch (e: Exception) {
            Log.e("Thumbnail", "Failed to generate thumbnail: ${e.message}")
            null
        }
    }


    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val item = items[position]

        if (!item.annotatedImageUrl.isNullOrBlank()) {
            // If an annotated image is available, load it with Glide
            Glide.with(context)
                .load(item.annotatedImageUrl)
                .placeholder(android.R.drawable.ic_menu_report_image)
                .into(holder.imageView)
        } else {
            // Otherwise, try to generate a thumbnail from the video
            val thumbnail = generateVideoThumbnail(item.videoUrl, (item.frameTime * 1000).toLong())
            if (thumbnail != null) {
                holder.imageView.setImageBitmap(thumbnail)
            } else {
                // Fallback: show a default placeholder if thumbnail generation fails
                holder.imageView.setImageResource(android.R.drawable.ic_menu_report_image)
            }
        }

        holder.itemView.setOnClickListener {
            val intent = Intent(context, VideoPlayerActivity::class.java)
            intent.putExtra("video_url", item.videoUrl)
            intent.putExtra("frame_time", item.frameTime)
            intent.putExtra("annotated_image", item.annotatedImageUrl)
            intent.putExtra("currentQuery", query)
            intent.putExtra("emotion", emotion)
            intent.putExtra("embedding_id", item.embeddingID)
            intent.putExtra("dataType", dataType)
            context.startActivity(intent)
        }
    }

    override fun getItemCount(): Int = items.size
}
