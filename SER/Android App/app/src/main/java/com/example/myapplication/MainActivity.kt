package com.example.myapplication
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.android.volley.Response
import com.android.volley.toolbox.StringRequest
import com.android.volley.toolbox.Volley
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.YuvImage
import android.util.Base64
import androidx.camera.core.ImageProxy
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import com.android.volley.*
import com.android.volley.toolbox.HttpHeaderParser


class MainActivity : AppCompatActivity() {

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var previewView: PreviewView
    private val REQUIRED_PERMISSIONS = arrayOf(android.Manifest.permission.CAMERA)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (allPermissionsGranted()) {
            startCameraStream()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS,
                Companion.REQUEST_CODE_PERMISSIONS
            )
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


    private fun startCameraStream() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            // front camera selection
            val cameraSelector = CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                .build()

            // TODO make image sending only every two seconds
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analyzer ->
                    analyzer.setAnalyzer(cameraExecutor) { image ->
                        Log.d("CameraStream", "Frame received: ${image.format}, ${image.width}x${image.height}")
                        val bitmap = imageProxyToBitmap(image)
                        sendPostRequest(this, bitmap)
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

    fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val imagePlanes = image.planes
        val buffer = imagePlanes[0].buffer // Y plane
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)

        val yuvImage = YuvImage(bytes, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, image.width, image.height), 100, out)
        val jpegBytes = out.toByteArray()

        return BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
    }

    fun bitmapToBase64(bitmap: Bitmap): String {
        val byteArrayOutputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream)
        val byteArray = byteArrayOutputStream.toByteArray()
        return Base64.encodeToString(byteArray, Base64.DEFAULT)
    }


    fun sendPostRequest(context: android.content.Context, bitmap: Bitmap) {
        val url = "http://192.168.103.178:8003/upload_base64" // Replace with your API URL
        val base64Image = bitmapToBase64(bitmap)

        val jsonBody = JSONObject()
        jsonBody.put("image", base64Image) // Make sure this matches FastAPI
        val requestBody = jsonBody.toString()

        val requestQueue: RequestQueue = Volley.newRequestQueue(context)

        val stringRequest = object : StringRequest(Method.POST, url,
            Response.Listener { response ->
                Log.i("VOLLEY", "Success! Response: $response")
            },
            Response.ErrorListener { error ->
                Log.e("VOLLEY", "Error: ${error.message}")
                if (error.networkResponse != null) {
                    val statusCode = error.networkResponse.statusCode
                    val responseData = error.networkResponse.data?.let { String(it) } ?: "No response body"
                    Log.e("VOLLEY", "HTTP Status Code: $statusCode")
                    Log.e("VOLLEY", "Response Data: $responseData")
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