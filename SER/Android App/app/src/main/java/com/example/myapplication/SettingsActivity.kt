package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.MenuItem
import android.widget.Switch
import androidx.appcompat.app.AppCompatActivity

class SettingsActivity : AppCompatActivity() {

    private lateinit var switchDuplicateVideos: Switch
    private var duplicateVideos = true;
    private lateinit var switchDarkMode: Switch
    private var darkMode = false;

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)  // Enables the back arrow

        // Duplicate Videos Switch
        switchDuplicateVideos = findViewById(R.id.duplicateVideos)
        switchDuplicateVideos.setOnCheckedChangeListener { _, isChecked ->
            duplicateVideos = isChecked
            Log.d("SWITCH", "allowed duplicate Videos $duplicateVideos")
        }

        // Dark Mode Switch
        switchDarkMode = findViewById(R.id.darkmode)
        switchDarkMode.setOnCheckedChangeListener { _, isChecked ->
            darkMode = isChecked
            Log.d("SWITCH", "allowed darkMode $darkMode")
        }
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            android.R.id.home -> {
                // Send results back to MainActivity when clicking on the back arrow
                val resultIntent = Intent()
                resultIntent.putExtra("allowDuplicateVideos", duplicateVideos)
                resultIntent.putExtra("darkMode", darkMode)
                setResult(RESULT_OK, resultIntent)
                finish()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }
}
