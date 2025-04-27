package com.example.myapplication

import android.content.Intent
import android.content.res.Configuration
import android.os.Bundle
import android.util.Log
import android.view.MenuItem
import android.view.View
import android.widget.CheckBox
import android.widget.LinearLayout
import android.widget.Switch
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate

class SettingsActivity : AppCompatActivity() {

    private lateinit var switchDuplicateVideos: Switch
    private var duplicateVideos = true;
    private lateinit var switchDarkMode: Switch
    private var darkMode = true;
    private lateinit var switchCheerupMode: Switch
    private var cheerupMode = false;
    private var jokesActivated = false;
    private var complimentsActivated = false;
    private lateinit var jokesCheckbox : CheckBox
    private lateinit var complimentsCheckbox : CheckBox

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)  // Enables the back arrow

        switchDarkMode = findViewById(R.id.darkmode)
        switchDuplicateVideos = findViewById(R.id.duplicateVideos)
        switchCheerupMode = findViewById(R.id.cheerupMode)
        jokesCheckbox = findViewById(R.id.jokeCheckbox)
        complimentsCheckbox = findViewById(R.id.complimentCheckbox)

        val sharedPref = getSharedPreferences("AppSettings", MODE_PRIVATE)
        darkMode = sharedPref.getBoolean("darkMode", false)
        switchDarkMode.isChecked = darkMode

        switchDuplicateVideos.setOnCheckedChangeListener { _, isChecked ->
            duplicateVideos = isChecked
            with(sharedPref.edit()) {
                putBoolean("duplicateVideos", duplicateVideos)
                apply()
            }
            Log.d("SWITCH", "allowed duplicate Videos $duplicateVideos")
        }

        // Cheer Up Mode
        cheerupMode = sharedPref.getBoolean("cheerupMode", false)
        jokesActivated = sharedPref.getBoolean("jokesActivated", false)
        complimentsActivated = sharedPref.getBoolean("complimentsActivated", false)
        switchCheerupMode.isChecked = cheerupMode
        jokesCheckbox.isChecked = jokesActivated
        complimentsCheckbox.isChecked = complimentsActivated

        switchCheerupMode.setOnCheckedChangeListener { _, isChecked ->
            cheerupMode = isChecked
            with(sharedPref.edit()) {
                putBoolean("cheerupMode", cheerupMode)
                apply()
            }
            Log.d("SWITCH", "allowed CheerupMode $cheerupMode")
        }


        val cheerupOptionsLayout = findViewById<LinearLayout>(R.id.cheerupOptionsLayout)

        switchCheerupMode.isChecked = cheerupMode
        cheerupOptionsLayout.visibility = if (cheerupMode) View.VISIBLE else View.GONE

        switchCheerupMode.setOnCheckedChangeListener { _, isChecked ->
            cheerupMode = isChecked
            with(sharedPref.edit()) {
                putBoolean("cheerupMode", cheerupMode)
                apply()
            }
            cheerupOptionsLayout.visibility = if (isChecked) View.VISIBLE else View.GONE
        }


        switchDarkMode.setOnCheckedChangeListener { _, isChecked ->
            darkMode = isChecked

            // Save preference
            with(sharedPref.edit()) {
                putBoolean("darkMode", darkMode)
                apply()
            }
        }
        fun validateCheerupOptions() {
            if (!jokesCheckbox.isChecked && !complimentsCheckbox.isChecked) {
                // Force at least one to be checked (default to jokes)
                jokesCheckbox.isChecked = true
                jokesActivated = true
                with(sharedPref.edit()) {
                    putBoolean("jokesActivated", jokesActivated)
                    apply()
                }
                Toast.makeText(this, "At least one option must be selected.", Toast.LENGTH_SHORT).show()
            }
        }
        jokesCheckbox.setOnCheckedChangeListener { _, isChecked ->
            jokesActivated = isChecked
            with(sharedPref.edit()) {
                putBoolean("jokesActivated", jokesActivated)
                apply()
            }
            validateCheerupOptions()
        }

        complimentsCheckbox.setOnCheckedChangeListener { _, isChecked ->
            complimentsActivated = isChecked
            with(sharedPref.edit()) {
                putBoolean("complimentsActivated", complimentsActivated)
                apply()
            }
            validateCheerupOptions()
        }

    }


    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            android.R.id.home -> {
                // Send results back to MainActivity when clicking on the back arrow
                val resultIntent = Intent()
                resultIntent.putExtra("allowDuplicateVideos", duplicateVideos)
                resultIntent.putExtra("darkMode", darkMode)
                resultIntent.putExtra("cheerupMode", cheerupMode)
                resultIntent.putExtra("jokesActivated", jokesActivated)
                resultIntent.putExtra("complimentsActivated", complimentsActivated)
                setResult(RESULT_OK, resultIntent)
                finish()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }
}
