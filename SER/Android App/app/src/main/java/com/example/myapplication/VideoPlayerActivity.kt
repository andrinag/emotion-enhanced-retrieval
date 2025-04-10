package com.example.myapplication

import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.net.Uri
import android.os.Bundle
import android.util.Base64
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.android.volley.DefaultRetryPolicy
import com.android.volley.NetworkResponse
import com.android.volley.Request
import com.android.volley.Request.Method
import com.android.volley.RequestQueue
import com.android.volley.Response
import com.android.volley.toolbox.HttpHeaderParser
import com.android.volley.toolbox.StringRequest
import com.android.volley.toolbox.Volley
import com.bumptech.glide.Glide
import com.example.myapplication.MainActivity
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.net.URLDecoder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class VideoPlayerActivity : AppCompatActivity() {

    private lateinit var videoView: VideoView
    private lateinit var progressBar: ProgressBar
    private lateinit var titleView: TextView
    private lateinit var checkbox: CheckBox
    private lateinit var imageView: ImageView
    private lateinit var cameraExecutor: ExecutorService
    private val REQUIRED_PERMISSIONS = arrayOf(android.Manifest.permission.CAMERA)
    var userEmotion: String = "happy"
    var expectingAnswerLlama = false
    var adaptedQuery = ""
    private lateinit var suggestionsRecyclerView: RecyclerView
    private lateinit var suggestionsAdapter: ResultsAdapter


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_video_player)

        videoView = findViewById(R.id.videoPlayerView)
        progressBar = findViewById(R.id.loadingSpinner)
        titleView = findViewById(R.id.videoTitle)
        checkbox = findViewById(R.id.checkboxShowAnnotation)
        imageView = findViewById(R.id.imageAnnotation)
        cameraExecutor = Executors.newSingleThreadExecutor()
        suggestionsRecyclerView = findViewById(R.id.suggestionsRecyclerView)
        suggestionsRecyclerView.layoutManager = LinearLayoutManager(this, LinearLayoutManager.HORIZONTAL, false)


        if (allPermissionsGranted()) {
            startCameraStream()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS,
                Companion.REQUEST_CODE_PERMISSIONS
            )
        }

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


    fun sendQueryRequestLlama(
        context: android.content.Context,
        query: String,
        emotionSpinner: String,
        callback: (JSONArray) -> Unit
    ) {
        val url = "http://10.34.64.139:8001/ask_llama/$query/$emotionSpinner"

        val requestQueue: RequestQueue = Volley.newRequestQueue(context)

        val stringRequest = StringRequest(
            Request.Method.GET, url,
            Response.Listener<String> { response ->
                Log.i("LLAMA", "Success! Response: $response")
                try {
                    val result = JSONArray(response)
                    val videoResults = mutableListOf<VideoResult>()
                    val baseUrl = "http://10.34.64.139:8001"

                    for (i in 0 until result.length()) {
                        val obj = result.getJSONObject(i)
                        val videoUrl = baseUrl + obj.getString("video_path")
                        val frameTime = obj.optDouble("frame_time", 0.0)
                        val annotatedImage = obj.optString("annotated_image", null)?.let { "$baseUrl/$it" }

                        videoResults.add(VideoResult(videoUrl, frameTime, annotatedImage))
                    }

                    suggestionsAdapter = ResultsAdapter(videoResults, this, query, emotionSpinner)
                    suggestionsRecyclerView.adapter = suggestionsAdapter

                } catch (e: Exception) {
                    Log.e("VOLLEY", "JSON Parsing Error: ${e.message}")
                }

            },
            { error ->
                Log.e("LLAMA", "Volley Error: ${error.message}")

                if (error.networkResponse != null) {
                    val statusCode = error.networkResponse.statusCode
                    val responseBody = error.networkResponse.data?.let { String(it) }
                    Log.e("LLAMA", "HTTP Status Code: $statusCode")
                    Log.e("LLAMA", "Error Response Body: $responseBody")
                }
            })

        stringRequest.retryPolicy = DefaultRetryPolicy(
            100000,
            DefaultRetryPolicy.DEFAULT_MAX_RETRIES,
            DefaultRetryPolicy.DEFAULT_BACKOFF_MULT
        )
        requestQueue.add(stringRequest)
    }

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
                    userEmotion = jsonResponse.optString("emotion", "Unknown")
                    if ((userEmotion == "sad" && !expectingAnswerLlama)|| (userEmotion == "angry" && !expectingAnswerLlama)) {
                        Log.d("LLAMA", "sending to llama")
                        expectingAnswerLlama = true

                        val query = intent.getStringExtra("currentQuery")
                        val emotionSpinner = intent.getStringExtra("emotion")
                        Log.d("VideoPlayerActivity", "Received query: $query")


                        if (query == null) {
                            Log.e("LLAMA", "Query is null, not sending to LLaMA")
                            return@Listener
                        }
                        if (emotionSpinner == null) {
                            Log.e("LLAMA", "Query is null, not sending to LLaMA")
                            return@Listener
                        }

                        sendQueryRequestLlama(this, query, emotionSpinner) { result ->
                            if (true) {
                                Log.d("LLAMA", "response: $result")
                                adaptedQuery = result.toString()
                                expectingAnswerLlama = false
                            } else {
                                Log.e("LLAMA", "Request failed")
                            }
                        }
                    }

                    runOnUiThread {
                        findViewById<TextView>(R.id.text).text = "Sentiment: $sentiment\nEmotion: $userEmotion"
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


    private var lastSentTime = 0L // stores the last time image was sent

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

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
    }

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

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }
}
