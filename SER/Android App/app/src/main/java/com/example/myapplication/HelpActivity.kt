package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.view.MenuItem
import androidx.appcompat.app.AppCompatActivity

class HelpActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_help)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)  // Enables the back arrow

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