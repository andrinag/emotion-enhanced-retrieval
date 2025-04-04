package com.example.myapplication

import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import com.android.volley.toolbox.StringRequest
import com.android.volley.toolbox.Volley
import java.util.concurrent.ExecutorService
import android.net.Uri
import android.util.Base64
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ImageView
import android.widget.MediaController
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.VideoView
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.android.volley.*
import com.android.volley.toolbox.HttpHeaderParser
import com.bumptech.glide.Glide
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var editTextQuery: EditText
    private lateinit var buttonSearch: Button
    private lateinit var simpleVideoView: VideoView
    lateinit var mediaControls: MediaController
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var progressBarVideo: ProgressBar
    private val REQUIRED_PERMISSIONS = arrayOf(android.Manifest.permission.CAMERA)
    private lateinit var spinnerDataType: android.widget.Spinner
    private lateinit var spinnerSentiment: android.widget.Spinner
    private lateinit var checkboxAnnotation: CheckBox
    private lateinit var imageView: ImageView


    /**
     * plays video as soon as activity is started
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        simpleVideoView = findViewById<View>(R.id.simpleVideoView) as VideoView
        editTextQuery = findViewById(R.id.editTextQuery)
        buttonSearch = findViewById(R.id.buttonSearch)
        mediaControls = MediaController(this)
        mediaControls.setAnchorView(simpleVideoView)
        progressBarVideo = findViewById(R.id.progressBarVideo)
        simpleVideoView.setMediaController(mediaControls)
        cameraExecutor = Executors.newSingleThreadExecutor()
        progressBarVideo.visibility = View.GONE
        spinnerDataType = findViewById(R.id.spinnerDataType)
        spinnerSentiment = findViewById(R.id.spinnerSentiment)
        val dataType = spinnerDataType.selectedItem.toString()
        val emotion = spinnerSentiment.selectedItem.toString()
        checkboxAnnotation = findViewById(R.id.checkboxShowAnnotation)
        imageView = findViewById<ImageView>(R.id.imageAnnotation)


        if (allPermissionsGranted()) {
            startCameraStream()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS,
                Companion.REQUEST_CODE_PERMISSIONS
            )
        }
        checkboxAnnotation.setOnCheckedChangeListener { _, isChecked ->
            imageView.visibility = if (isChecked) View.VISIBLE else View.GONE
        }

        buttonSearch.setOnClickListener {
            progressBarVideo.visibility = View.VISIBLE
            simpleVideoView.visibility = View.GONE
            val query = editTextQuery.text.toString().trim()
            if (query.isNotEmpty()) {
                sendPostRequestSentimentQuery(this, query)
                sendQueryRequestWithSentiment(this, query, dataType, emotion) { result ->
                    if (true) {
                        simpleVideoView.visibility = View.VISIBLE
                        Log.d("VOLLEY", "response: $result")
                    } else {
                        Log.e("VOLLEY", "Request failed")
                    }
                }
            }
        }
    }

    fun sendQueryRequestWithSentiment(
        context: android.content.Context,
        query: String,
        dataType: String,
        emotion: String,
        callback: (JSONArray) -> Unit
    ) {
        val url = "http://10.34.64.139:8001/search_combined_face/$query/$emotion"

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
                        val timeString = result.getJSONObject(0).getString("frame_time")
                        var time: Double = 0.0
                        try {
                            time = timeString.toDouble()
                        } catch (nfe: NumberFormatException) {
                            Log.e("VIDEO", "Could not convert timeString to Float")
                        }
                        val videoPath = firstVideo.getString("video_path")
                        val baseUrl = "http://10.34.64.139:8001"
                        val videoUrl = "$baseUrl$videoPath"
                        Log.d("VOLLEY", "Playing video from URL: $videoUrl")

                        val intent = Intent(this, SearchResultsActivity::class.java)
                        intent.putExtra("results_json", result.toString()) // send JSON as String
                        startActivity(intent)

                        progressBarVideo.visibility = View.GONE
                        (context as? MainActivity)?.playVideo(videoUrl, time)
                        playVideo(videoUrl, time)
                        val imagePath = firstVideo.optString("annotated_image", "")
                        if (imagePath.isNotEmpty()) {
                            val imageUrl = "http://10.34.64.139:8001/$imagePath"
                            Log.d("Image Path: ", imageUrl)
                            imageView.visibility = if (checkboxAnnotation.isChecked) View.VISIBLE else View.GONE

                            Glide.with(this)
                                .load(imageUrl)
                                .placeholder(android.R.drawable.ic_menu_report_image)
                                .into(imageView)
                        }
                    } else {
                        Log.e("VOLLEY", "No videos found in response")
                    }

                } catch (e: Exception) {
                    Log.e("VOLLEY", "JSON Parsing Error: ${e.message}")
                }
            },
            { error ->
                Log.e("VOLLEY", "Volley Error: ${error.message}")
                progressBarVideo.visibility = View.GONE

                if (error.networkResponse != null) {
                    val statusCode = error.networkResponse.statusCode
                    val responseBody = error.networkResponse.data?.let { String(it) }
                    Log.e("VOLLEY", "HTTP Status Code: $statusCode")
                    Log.e("VOLLEY", "Error Response Body: $responseBody")
                }
            })

        stringRequest.retryPolicy = DefaultRetryPolicy(
            100000,
            DefaultRetryPolicy.DEFAULT_MAX_RETRIES,
            DefaultRetryPolicy.DEFAULT_BACKOFF_MULT
        )
        progressBarVideo.visibility = View.VISIBLE
        requestQueue.add(stringRequest)
    }

    fun sendQueryRequest(
        context: android.content.Context,
        query: String,
        callback: (JSONArray) -> Unit
    ) {
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
                        val timeString = result.getJSONObject(0).getString("frame_time")
                        var time: Double = 0.0
                        try {
                            time = timeString.toDouble()
                        } catch (nfe: NumberFormatException) {
                            Log.e("VIDEO", "Could not convert timeString to Float")
                        }
                        val videoPath = firstVideo.getString("video_path")
                        val baseUrl = "http://10.34.64.139:8001"
                        val videoUrl = "$baseUrl$videoPath"
                        Log.d("VOLLEY", "Playing video from URL: $videoUrl")
                        progressBarVideo.visibility = View.GONE
                        (context as? MainActivity)?.playVideo(videoUrl, time)
                        playVideo(videoUrl, time)
                    } else {
                        Log.e("VOLLEY", "No videos found in response")
                    }

                } catch (e: Exception) {
                    Log.e("VOLLEY", "JSON Parsing Error: ${e.message}")
                }
            },
            { error ->
                Log.e("VOLLEY", "Volley Error: ${error.message}")
                progressBarVideo.visibility = View.GONE

                if (error.networkResponse != null) {
                    val statusCode = error.networkResponse.statusCode
                    val responseBody = error.networkResponse.data?.let { String(it) }
                    Log.e("VOLLEY", "HTTP Status Code: $statusCode")
                    Log.e("VOLLEY", "Error Response Body: $responseBody")
                }
            })

        stringRequest.retryPolicy = DefaultRetryPolicy(
            100000,
            DefaultRetryPolicy.DEFAULT_MAX_RETRIES,
            DefaultRetryPolicy.DEFAULT_BACKOFF_MULT
        )
        progressBarVideo.visibility = View.VISIBLE
        requestQueue.add(stringRequest)
    }


    private fun playVideo(url: String, time: Double) {
        Log.i("VIDEO", "Starting video playback for: $url")

        val uri = Uri.parse(url)
        simpleVideoView.setVideoURI(uri)

        simpleVideoView.setOnPreparedListener { mediaPlayer ->
            progressBarVideo.visibility = View.GONE
            simpleVideoView.visibility = View.VISIBLE
            val seekPositionMs = (time * 1000).toInt()
            mediaPlayer.seekTo(seekPositionMs)
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
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == Companion.REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCameraStream()
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    private var lastSentTime = 0L // stores the last time image was sent

    /**
     * starts the camera stream, sends image every 2s to the api
     */
    private fun startCameraStream() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
            val cameraSelector = CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                .build()

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analyzer ->
                    analyzer.setAnalyzer(cameraExecutor) { image ->
                        val currentTime = System.currentTimeMillis()

                        if (currentTime - lastSentTime >= 1000) {
                            Log.d("CameraStream", "Sending image...")

                            // val bitmap = imageProxyToBitmap(image)
                            val bitmap = imageProxyToBase64(image)
                            var response = sendPostRequestSentiment(this, bitmap)
                            lastSentTime = currentTime
                        }
                        image.close()
                    }
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
                Toast.makeText(this, "Front Camera Streaming Started!", Toast.LENGTH_SHORT).show()
            } catch (exc: Exception) {
                Log.e("CameraStream", "Failed to bind camera", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }


    /**
     * shuts down the camera when app is closed
     */
    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
    }


    /**
     * converts the image proxy to base64 representation such that it can be sent to sentiment api
     */
    fun imageProxyToBase64(image: ImageProxy): String {
        val yBuffer = image.planes[0].buffer // Y
        val vuBuffer = image.planes[2].buffer // VU
        val ySize = yBuffer.remaining()
        val vuSize = vuBuffer.remaining()
        val nv21 = ByteArray(ySize + vuSize)
        yBuffer.get(nv21, 0, ySize)
        vuBuffer.get(nv21, ySize, vuSize)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 50, out)
        val imageBytes = out.toByteArray()
        val bitmap =  BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

        val byteArrayOutputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream)
        val byteArray = byteArrayOutputStream.toByteArray()
        return Base64.encodeToString(byteArray, Base64.DEFAULT)
    }


    /**
     * sends post request (with VOLLEY) to the sentiment api
     */
    fun sendPostRequestSentiment(context: android.content.Context, base64Image: String) {
        // TODO needs to be changed to the node adress
        val url = "http://10.34.64.139:8003/upload_base64" // adress of the sentiment api
        val jsonBody = JSONObject()
        jsonBody.put("image", base64Image)
        val requestBody = jsonBody.toString()

        val requestQueue: RequestQueue = Volley.newRequestQueue(context)

        val stringRequest = object : StringRequest(Method.POST, url,
            Response.Listener { response ->
                Log.i("VOLLEY", "Success! Response: $response")

                try {
                    val jsonResponse = JSONObject(response)
                    val sentiment = jsonResponse.optString("sentiment", "Unknown")
                    val emotion = jsonResponse.optString("emotion", "Unknown")

                    runOnUiThread {
                        findViewById<TextView>(R.id.text).text = "Sentiment: $sentiment\nEmotion: $emotion"
                    }

                } catch (e: Exception) {
                    Log.e("VOLLEY", "Error parsing JSON response: ${e.message}")
                }

            },
            Response.ErrorListener { error ->
                Log.e("VOLLEY", "Error: ${error.message}")
                if (error.networkResponse != null) {
                    val statusCode = error.networkResponse.statusCode
                    val responseData = error.networkResponse.data?.let { String(it) } ?: "No response body"
                    Log.e("VOLLEY", "HTTP Status Code: $statusCode")
                    Log.e("VOLLEY", "Response Data: $responseData")
                }

                runOnUiThread {
                    findViewById<TextView>(R.id.text).text = "Error: Could not get response"
                }
            }) {

            override fun getBodyContentType(): String {
                return "application/json; charset=utf-8"
            }

            override fun getBody(): ByteArray {
                return requestBody.toByteArray(Charsets.UTF_8)
            }

            override fun parseNetworkResponse(response: NetworkResponse?): Response<String> {
                val responseString = response?.data?.let { String(it) } ?: "No Response"
                return Response.success(responseString, HttpHeaderParser.parseCacheHeaders(response))
            }
        }

        requestQueue.add(stringRequest)
    }

    fun sendPostRequestSentimentQuery(context: android.content.Context, query: String) {
        // TODO needs to be changed to the node adress
        val url = "http://10.34.64.139:8003/upload_query" // adress of the sentiment api
        val jsonBody = JSONObject()
        jsonBody.put("query", query)
        val requestBody = jsonBody.toString()

        val requestQueue: RequestQueue = Volley.newRequestQueue(context)

        val stringRequest = object : StringRequest(Method.POST, url,
            Response.Listener { response ->
                Log.i("VOLLEY", "Success! Response: $response")

                try {
                    val jsonResponse = JSONObject(response).getJSONArray("emotion")
                    val firstSentiment = jsonResponse.getJSONObject(0)
                    val sentimentLabel = firstSentiment.getString("label")
                    val sentimentScore = firstSentiment.getDouble("score")

                    val sentimentText = "Sentiment: $sentimentLabel (${String.format("%.2f", sentimentScore)})"
                    runOnUiThread {
                        findViewById<TextView>(R.id.querySentiment).text = sentimentText
                    }

                } catch (e: Exception) {
                    Log.e("VOLLEY", "Error parsing JSON response: ${e.message}")
                }

            },
            Response.ErrorListener { error ->
                Log.e("VOLLEY", "Error: ${error.message}")
                if (error.networkResponse != null) {
                    val statusCode = error.networkResponse.statusCode
                    val responseData = error.networkResponse.data?.let { String(it) } ?: "No response body"
                    Log.e("VOLLEY", "HTTP Status Code: $statusCode")
                    Log.e("VOLLEY", "Response Data: $responseData")
                }

                runOnUiThread {
                    findViewById<TextView>(R.id.querySentiment).text = "Error: Could not get response"
                }
            }) {

            override fun getBodyContentType(): String {
                return "application/json; charset=utf-8"
            }

            override fun getBody(): ByteArray {
                return requestBody.toByteArray(Charsets.UTF_8)
            }

            override fun parseNetworkResponse(response: NetworkResponse?): Response<String> {
                val responseString = response?.data?.let { String(it) } ?: "No Response"
                return Response.success(responseString, HttpHeaderParser.parseCacheHeaders(response))
            }
        }

        requestQueue.add(stringRequest)
    }
}