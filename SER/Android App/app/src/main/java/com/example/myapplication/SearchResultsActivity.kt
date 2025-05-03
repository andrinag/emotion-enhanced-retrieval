package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.MenuItem
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import org.json.JSONArray


/**
 * Displays a scrollable list of video search results (with optional annotated image).
 * Activity is launched after the user submits search query in MainActivity.
 * Receives a JSON string via Intent and parses it into a list of VideoResults items and displays
 * each result in a RecycleView using ResultsAdapter.
 */
class SearchResultsActivity : AppCompatActivity() {

    private lateinit var recyclerView: RecyclerView
    private lateinit var adapter: ResultsAdapter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_search_results)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)  // Enables the back arrow

        recyclerView = findViewById(R.id.resultsRecyclerView)
        recyclerView.layoutManager = LinearLayoutManager(this)

        val jsonString = intent.getStringExtra("results_json")
        val jsonArray = JSONArray(jsonString)
        val baseUrl = "http://10.34.64.139:8001"

        val videoResults = mutableListOf<VideoResult>()
        for (i in 0 until jsonArray.length()) {
            val obj = jsonArray.getJSONObject(i)
            val videoUrl = baseUrl + obj.getString("video_path")
            Log.d("VIDEO", "received video path$videoUrl")
            val frameTime = obj.optString("frame_time", "0.0").toDoubleOrNull() ?: 0.0
            val embeddingID = obj.optInt("embedding_id", -1)
            val previousEmbeddingID = obj.optInt("previous_embedding_id", -1)
            var annotatedImage = obj.optString("annotated_image", "")
            var frameLocation = obj.optString("frame_location", "")
            annotatedImage = if (annotatedImage.isNotBlank()) "$baseUrl/$annotatedImage" else ""
            frameLocation = if (frameLocation.isNotBlank()) "$baseUrl/$frameLocation" else ""
            Log.d("VIDEO", "received annotated image path$annotatedImage")
            Log.d("VIDEO", "received frame location image path $frameLocation")
            videoResults.add(
                VideoResult(
                    videoUrl,
                    frameTime,
                    annotatedImage,
                    frameLocation,
                    embeddingID,
                    previousEmbeddingID
                )
            )
        }
        Log.d("SearchResultsActivity", "Parsed ${videoResults.size} results")

        val query = intent.getStringExtra("currentQuery") ?: ""
        val emotionSpinner = intent.getStringExtra("emotion") ?: ""
        val dataType = intent.getStringExtra("dataType") ?: ""
        val suggestionMode = intent.getStringExtra("suggestionMode") ?: "nearest"
        val duplicateVideos = intent.getBooleanExtra("duplicateVideos", true)
        Log.d("Query", "Query in Search Result Acitivity is $query")
        adapter = ResultsAdapter(
            videoResults, this, query, emotionSpinner, dataType,
            suggestionMode, duplicateVideos
        ) // pass the query to adapter

        recyclerView.adapter = adapter
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            android.R.id.home -> {
                // Send results back to MainActivity when clicking on the back arrow
                val resultIntent = Intent()
                setResult(RESULT_OK, resultIntent)
                finish()
                true
            }

            else -> super.onOptionsItemSelected(item)
        }
    }
}
