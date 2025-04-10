package com.example.myapplication

import android.os.Bundle
import android.util.Log
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
            val annotatedImage = obj.optString("annotated_image", null)?.let { "$baseUrl/$it" }
            Log.d("VIDEO", "received annotated image path$annotatedImage")
            videoResults.add(VideoResult(videoUrl, frameTime, annotatedImage))
        }
        Log.d("SearchResultsActivity", "Parsed ${videoResults.size} results")

        val query = intent.getStringExtra("currentQuery") ?: ""
        val emotionSpinner = intent.getStringExtra("emotion") ?: ""
        Log.d("Query", "Query in Search Result Acitivity is $query")
        adapter = ResultsAdapter(videoResults, this, query, emotionSpinner) // pass the query to adapter

        recyclerView.adapter = adapter
    }
}
