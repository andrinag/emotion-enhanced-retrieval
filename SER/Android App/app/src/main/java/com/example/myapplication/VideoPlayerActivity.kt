package com.example.myapplication

import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import com.bumptech.glide.Glide
import java.net.URLDecoder

class VideoPlayerActivity : AppCompatActivity() {

    private lateinit var videoView: VideoView
    private lateinit var progressBar: ProgressBar
    private lateinit var titleView: TextView
    private lateinit var checkbox: CheckBox
    private lateinit var imageView: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_video_player)

        videoView = findViewById(R.id.videoPlayerView)
        progressBar = findViewById(R.id.loadingSpinner)
        titleView = findViewById(R.id.videoTitle)
        checkbox = findViewById(R.id.checkboxShowAnnotation)
        imageView = findViewById(R.id.imageAnnotation)

        val videoUrl = intent.getStringExtra("video_url") ?: run {
            Toast.makeText(this, "Missing video URL", Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        val frameTime = intent.getDoubleExtra("frame_time", 0.0)
        val imageUrl = intent.getStringExtra("annotated_image")

        // Set filename as title
        val filename = Uri.parse(videoUrl).lastPathSegment ?: "Unknown Video"
        titleView.text = URLDecoder.decode(filename, "UTF-8")

        // Load annotated image if present
        if (!imageUrl.isNullOrBlank()) {
            Glide.with(this)
                .load(imageUrl)
                .placeholder(android.R.drawable.ic_menu_report_image)
                .into(imageView)
        }

        checkbox.setOnCheckedChangeListener { _, isChecked ->
            imageView.visibility = if (isChecked && !imageUrl.isNullOrBlank()) View.VISIBLE else View.GONE
        }

        val mediaController = MediaController(this)
        mediaController.show(0)
        mediaController.setAnchorView(videoView)
        videoView.setMediaController(mediaController)
        videoView.setVideoURI(Uri.parse(videoUrl))
        videoView.requestFocus()

        videoView.setOnPreparedListener { mp ->
            progressBar.visibility = ProgressBar.GONE
            val seekToMs = (frameTime * 1000).toInt()
            mp.seekTo(seekToMs)
            mp.start()
        }

        videoView.setOnCompletionListener {
            Toast.makeText(this, "Playback finished", Toast.LENGTH_SHORT).show()
        }

        videoView.setOnErrorListener { _, what, extra ->
            Log.e("VideoPlayer", "Error: what=$what, extra=$extra")
            Toast.makeText(this, "Error playing video", Toast.LENGTH_SHORT).show()
            true
        }
    }
}
