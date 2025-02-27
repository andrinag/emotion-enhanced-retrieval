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
import android.widget.Button
import android.widget.EditText
import android.widget.MediaController
import android.widget.VideoView
import com.android.volley.*
import org.json.JSONArray

class VideoActivity : AppCompatActivity() {

    private lateinit var editTextQuery: EditText
    private lateinit var buttonSearch: Button
    private lateinit var simpleVideoView: VideoView
    lateinit var mediaControls: MediaController


    /**
     * plays video as soon as activity is started
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.video_player)
        simpleVideoView = findViewById<View>(R.id.simpleVideoView) as VideoView
        editTextQuery = findViewById(R.id.editTextQuery)
        buttonSearch = findViewById(R.id.buttonSearch)
        mediaControls = MediaController(this)
        mediaControls.setAnchorView(simpleVideoView)
        simpleVideoView.setMediaController(mediaControls)

        buttonSearch.setOnClickListener {
            val query = editTextQuery.text.toString().trim()
            if (query.isNotEmpty()) {
                sendQueryRequest(this, query) { result ->
                    if (true) {
                        Log.d("VOLLEY", "response: $result")
                    } else {
                        Log.e("VOLLEY", "Request failed")
                    }
                }
            }
        }
    }


    fun sendQueryRequest(context: android.content.Context, query: String, callback: (JSONArray) -> Unit) {
        val url = "http://10.34.64.139:8001/search/$query"

        val requestQueue: RequestQueue = Volley.newRequestQueue(context)

        val stringRequest = StringRequest(
            Request.Method.GET, url,
            Response.Listener<String> { response ->
                Log.i("VOLLEY", "Success! Response: $response")

                try {
                    val result = JSONArray(response)
                    Log.d("VOLLEY", "Callback being executed with response: $result")
                    callback(result)
                    Log.i("VIDEO", "starting to play the video here")
                    if (result.length() > 0) {
                        val firstVideo = result.getJSONObject(0)
                        val videoPath = firstVideo.getString("video_path")
                        val baseUrl = "http://10.34.64.139:8001"
                        val videoUrl = "$baseUrl$videoPath"
                        Log.d("VOLLEY", "Playing video from URL: $videoUrl")
                        (context as? VideoActivity)?.playVideo(videoUrl)
                        playVideo(videoUrl)
                    } else {
                        Log.e("VOLLEY", "No videos found in response")
                    }

                } catch (e: Exception) {
                    Log.e("VOLLEY", "JSON Parsing Error: ${e.message}")
                }
            },
            { error ->
                Log.e("VOLLEY", "Volley Error: ${error.message}")

                if (error.networkResponse != null) {
                    val statusCode = error.networkResponse.statusCode
                    val responseBody = error.networkResponse.data?.let { String(it) }
                    Log.e("VOLLEY", "HTTP Status Code: $statusCode")
                    Log.e("VOLLEY", "Error Response Body: $responseBody")
                }
            })

        stringRequest.retryPolicy = DefaultRetryPolicy(
            10000,
            DefaultRetryPolicy.DEFAULT_MAX_RETRIES,
            DefaultRetryPolicy.DEFAULT_BACKOFF_MULT
        )

        requestQueue.add(stringRequest)
    }


    private fun playVideo(url: String) {
        Log.i("VIDEO", "Starting video playback for: $url")

        val uri = Uri.parse(url)
        simpleVideoView.setVideoURI(uri)

        simpleVideoView.setOnPreparedListener { mediaPlayer ->
            mediaPlayer.start()
        }

        simpleVideoView.setOnCompletionListener {
            Log.i("VIDEO", "Playback completed")
            Toast.makeText(this, "Video completed", Toast.LENGTH_LONG).show()
        }

        simpleVideoView.setOnErrorListener { _, _, _ ->
            Log.e("VIDEO", "Error playing video: $url")
            Toast.makeText(this, "Error playing video", Toast.LENGTH_LONG).show()
            false
        }
    }
}