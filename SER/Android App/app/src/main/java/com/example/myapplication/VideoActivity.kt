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

        sendQueryRequest(this, "cat") { result ->
            if (result != null) {
                Log.d("VOLLEY", "response: $result" )
            } else {
                Log.e("VOLLEY", "Request failed")
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


    fun playVideo(url: String) {
        if (!::mediaControls.isInitialized) {
            mediaControls = MediaController(this)
            mediaControls.setAnchorView(this.simpleVideoView)
        }
        simpleVideoView.setMediaController(mediaControls)
        simpleVideoView.setVideoURI(Uri.parse(url))

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