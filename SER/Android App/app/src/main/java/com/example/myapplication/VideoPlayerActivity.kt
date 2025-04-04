package com.example.myapplication

import android.net.Uri
import android.os.Bundle
import android.widget.MediaController
import android.widget.ProgressBar
import android.widget.Toast
import android.widget.VideoView
import androidx.appcompat.app.AppCompatActivity


/**
 * VideoPlayerActivity that plays only one video.
 */
class VideoPlayerActivity : AppCompatActivity() {

    private lateinit var videoView: VideoView
    private lateinit var progressBar: ProgressBar

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_video_player)

        videoView = findViewById(R.id.videoPlayerView)
        progressBar = findViewById(R.id.loadingSpinner)

        val videoUrl = intent.getStringExtra("video_url") ?: return
        val frameTime = intent.getDoubleExtra("frame_time", 0.0)

        val mediaController = MediaController(this)
        mediaController.setAnchorView(videoView)
        videoView.setMediaController(mediaController)

        videoView.setVideoURI(Uri.parse(videoUrl))
        videoView.requestFocus()

        videoView.setOnPreparedListener { mp ->
            progressBar.visibility = ProgressBar.GONE
            mp.seekTo((frameTime * 1000).toInt())
            mp.start()
        }

        videoView.setOnCompletionListener {
            Toast.makeText(this, "Video finished", Toast.LENGTH_SHORT).show()
        }

        videoView.setOnErrorListener { _, _, _ ->
            Toast.makeText(this, "Error playing video", Toast.LENGTH_SHORT).show()
            true
        }
    }
}
