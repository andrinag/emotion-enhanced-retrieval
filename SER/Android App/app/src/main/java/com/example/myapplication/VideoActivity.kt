package com.example.myapplication

import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import com.android.volley.toolbox.StringRequest
import com.android.volley.toolbox.Volley
import java.util.concurrent.ExecutorService
import android.net.Uri
import android.view.View
import android.widget.MediaController
import android.widget.VideoView
import com.android.volley.*
import org.json.JSONArray

class VideoActivity : AppCompatActivity() {

    lateinit var simpleVideoView: VideoView
    lateinit var mediaControls: MediaController

    /**
     * plays video as soon as activity is started
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.video_player)
        simpleVideoView = findViewById<View>(R.id.simpleVideoView) as VideoView
        playVideo()

        // sendQueryRequest(this, "cat")
    }


    fun sendQueryRequest(context: android.content.Context, query: String) {
        val url = "http://10.34.64.139:8001/search/$query"

        val requestQueue: RequestQueue = Volley.newRequestQueue(context)

        val stringRequest = StringRequest(
            Request.Method.GET, url,
            { response ->
                Log.i("VOLLEY", "Success! Response: $response")
                val jsonArray = JSONArray(response)
            },
            { error ->
                Log.e("VOLLEY", "Error: ${error.message}")
            })

        requestQueue.add(stringRequest)
    }


    fun playVideo() {
        if (!::mediaControls.isInitialized) {
            mediaControls = MediaController(this)
            mediaControls.setAnchorView(this.simpleVideoView)
        }
        simpleVideoView.setMediaController(mediaControls)
        simpleVideoView.setVideoURI(Uri.parse("android.resource://" + packageName + "/" + R.raw.giraffes_1280p))

        simpleVideoView.requestFocus()
        simpleVideoView.start()
        simpleVideoView.setOnCompletionListener {
            Toast.makeText(
                applicationContext, "Video completed",
                Toast.LENGTH_LONG
            ).show()
            true
        }
        simpleVideoView.setOnErrorListener { mp, what, extra ->
            Toast.makeText(
                applicationContext,
                "An Error Occurred " + "While Playing Video !!!",
                Toast.LENGTH_LONG
            ).show()
            false
        }
    }

}