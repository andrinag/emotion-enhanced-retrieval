package com.example.myapplication

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/**
 * Activity that is started, whenever a user clicks on the "SEE UPDATED QUERY" button
 * in the video display when the LLM-suggestion-mode is activated.
 */
class LlamaQueryActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_llama_query)

        supportActionBar?.setDisplayHomeAsUpEnabled(true)

        val queryTextView = findViewById<TextView>(R.id.llamaQueryTextView)
        val llamaQuery = intent.getStringExtra("llama_query") ?: "No query provided."

        queryTextView.text = llamaQuery
    }

    override fun onSupportNavigateUp(): Boolean {
        finish()
        return true
    }
}
