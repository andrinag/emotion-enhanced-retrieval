package com.example.myapplication

import android.content.Context
import android.net.Uri
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.VideoView
import androidx.recyclerview.widget.RecyclerView
import com.bumptech.glide.Glide

/**
 * Recycle View Adapter for displaying a list of video results with optional annotated image.
 * Every item of the list contains a VideoView with start time and optionally shown image (face, ocr).
 */
class ResultsAdapter(
    private val items: List<VideoResult>,
    private val context: Context
) : RecyclerView.Adapter<ResultsAdapter.ViewHolder>() {


    /**
     * ViewHolder for one item in the RecycleView
     */
    inner class ViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val videoView: VideoView = itemView.findViewById(R.id.videoItemView)
        val imageView: ImageView = itemView.findViewById(R.id.annotationImage)
    }

    /**
     * Creates another ViewHolder
     */
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.result_item, parent, false)
        return ViewHolder(view)
    }

    /**
     * Binds a VideoResult to the ViewHolder at a given Position (time)
     */
    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val item = items[position]
        holder.videoView.setVideoURI(Uri.parse(item.videoUrl))
        holder.videoView.setOnPreparedListener { mediaPlayer ->
            mediaPlayer.seekTo((item.frameTime * 1000).toInt())
            mediaPlayer.start()
        }

        item.annotatedImageUrl?.let {
            Glide.with(context)
                .load(it)
                .into(holder.imageView)
        }
    }

    override fun getItemCount(): Int = items.size
}
