package com.example.myapplication

import android.content.Context
import android.content.Intent
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
    private val context: Context
) : RecyclerView.Adapter<ResultsAdapter.ViewHolder>() {

    inner class ViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val imageView: ImageView = itemView.findViewById(R.id.annotationImage)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.result_item, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val item = items[position]

        // Load annotated image or fallback placeholder (e.g. for asr)
        val imageUrl = item.annotatedImageUrl
            ?: "https://via.placeholder.com/640x360.png?text=No+Thumbnail"

        Glide.with(context)
            .load(imageUrl)
            .placeholder(android.R.drawable.ic_menu_report_image)
            .into(holder.imageView)

        // open the actual videoplayer when clicked
        holder.itemView.setOnClickListener {
            val intent = Intent(context, VideoPlayerActivity::class.java)
            intent.putExtra("video_url", item.videoUrl)
            intent.putExtra("frame_time", item.frameTime)
            intent.putExtra("annotated_image", item.annotatedImageUrl)
            context.startActivity(intent)
        }

    }

    override fun getItemCount(): Int = items.size
}
