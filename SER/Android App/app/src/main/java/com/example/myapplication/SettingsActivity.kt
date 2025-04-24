package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.MenuItem
import android.widget.Switch
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate

class SettingsActivity : AppCompatActivity() {

    private lateinit var switchDuplicateVideos: Switch
    private var duplicateVideos = true;
    private lateinit var switchDarkMode: Switch
    private var darkMode = true;

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)  // Enables the back arrow

        switchDarkMode = findViewById(R.id.darkmode)
        switchDuplicateVideos = findViewById(R.id.duplicateVideos)

        val sharedPref = getSharedPreferences("AppSettings", MODE_PRIVATE)
        darkMode = sharedPref.getBoolean("darkMode", false)
        switchDarkMode.isChecked = darkMode

        switchDuplicateVideos.setOnCheckedChangeListener { _, isChecked ->
            duplicateVideos = isChecked
            Log.d("SWITCH", "allowed duplicate Videos $duplicateVideos")
        }

        switchDarkMode.setOnCheckedChangeListener { _, isChecked ->
            darkMode = isChecked

            // Save preference
            with(sharedPref.edit()) {
                putBoolean("darkMode", darkMode)
                apply()
            }
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
