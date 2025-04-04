package com.example.myapplication

import android.content.Context
import android.net.Uri
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.MediaController
import android.widget.Toast
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
        val mediaController: MediaController = MediaController(itemView.context)
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
        val videoUri = Uri.parse(item.videoUrl)

        holder.videoView.setVideoURI(videoUri)

        holder.videoView.setOnPreparedListener { mediaPlayer ->
            val seekToMs = (item.frameTime * 1000).toInt()
            holder.videoView.seekTo(seekToMs)

            // Attach MediaController AFTER prepared, else there is error playing the video!
            val mediaController = MediaController(context)
            mediaController.setAnchorView(holder.videoView)
            holder.videoView.setMediaController(mediaController)
        }

        holder.videoView.setOnCompletionListener {
            Log.i("VIDEO", "Playback completed")
        }

        holder.videoView.setOnErrorListener { _, what, extra ->
            Log.e("VIDEO", "Error playing video: $videoUri - what: $what extra: $extra")
            Toast.makeText(context, "Error playing video", Toast.LENGTH_SHORT).show()
            true
        }

        item.annotatedImageUrl?.let {
            Glide.with(context)
                .load(it)
                .into(holder.imageView)
            holder.imageView.visibility = View.VISIBLE
        } ?: run {
            holder.imageView.visibility = View.GONE
        }
    }


    override fun getItemCount(): Int = items.size
}
