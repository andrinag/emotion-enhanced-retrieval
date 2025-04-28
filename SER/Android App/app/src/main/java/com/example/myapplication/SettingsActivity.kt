package com.example.myapplication

import android.content.Intent
import android.content.res.Configuration
import android.os.Bundle
import android.util.Log
import android.view.MenuItem
import android.view.View
import android.widget.CheckBox
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
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
    private lateinit var suggestionsRadioGroup: RadioGroup
    private lateinit var radioLLM: RadioButton
    private lateinit var radioNearestNeighbor: RadioButton
    private lateinit var radioNone: RadioButton
    private var suggestionMode: String = "nearest"


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)  // Enables the back arrow

        switchDarkMode = findViewById(R.id.darkmode)
        switchDuplicateVideos = findViewById(R.id.duplicateVideos)
        switchCheerupMode = findViewById(R.id.cheerupMode)
        jokesCheckbox = findViewById(R.id.jokeCheckbox)
        complimentsCheckbox = findViewById(R.id.complimentCheckbox)
        suggestionsRadioGroup = findViewById(R.id.suggestionsRadioGroup)
        radioLLM = findViewById(R.id.radioLLM)
        radioNearestNeighbor = findViewById(R.id.radioNearestNeighbor)
        radioNone = findViewById(R.id.radioNone)


        val sharedPref = getSharedPreferences("AppSettings", MODE_PRIVATE)
        darkMode = sharedPref.getBoolean("darkMode", false)
        duplicateVideos = sharedPref.getBoolean("duplicateVideos", true)
        switchDuplicateVideos.isChecked = duplicateVideos
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
        suggestionMode = sharedPref.getString("suggestionMode", "nearest") ?: "nearest"
        when (suggestionMode) {
            "llm" -> radioLLM.isChecked = true
            "nearest" -> radioNearestNeighbor.isChecked = true
            "none" -> radioNone.isChecked = true
        }

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

        // suggestion Mode
        suggestionsRadioGroup.setOnCheckedChangeListener { _, checkedId ->
            suggestionMode = when (checkedId) {
                R.id.radioLLM -> "llm"
                R.id.radioNearestNeighbor -> "nearest"
                R.id.radioNone -> "none"
                else -> "nearest"
            }
            with(sharedPref.edit()) {
                putString("suggestionMode", suggestionMode)
                apply()
            }
            Log.d("RADIO", "Selected suggestion mode: $suggestionMode")
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
                resultIntent.putExtra("suggestionMode", suggestionMode)
                setResult(RESULT_OK, resultIntent)
                finish()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }
}
