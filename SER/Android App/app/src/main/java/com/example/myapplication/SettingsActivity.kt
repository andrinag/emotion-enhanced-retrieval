package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.MenuItem
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity

class SettingsActivity : AppCompatActivity() {

    private lateinit var switchDuplicateVideos: Switch
    private lateinit var switchDarkMode: Switch
    private lateinit var suggestionsRadioGroup: RadioGroup
    private lateinit var radioLLM: RadioButton
    private lateinit var radioNearestNeighbor: RadioButton
    private lateinit var radioNone: RadioButton

    private var duplicateVideos = true
    private var darkMode = true
    private var suggestionMode: String = "nearest"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)

        initViews()
        loadPreferences()
        setupUI()
    }

    private fun initViews() {
        switchDarkMode = findViewById(R.id.darkmode)
        switchDuplicateVideos = findViewById(R.id.duplicateVideos)
        suggestionsRadioGroup = findViewById(R.id.suggestionsRadioGroup)
        radioLLM = findViewById(R.id.radioLLM)
        radioNearestNeighbor = findViewById(R.id.radioNearestNeighbor)
        radioNone = findViewById(R.id.radioNone)
    }

    private fun loadPreferences() {
        val sharedPref = getSharedPreferences("AppSettings", MODE_PRIVATE)
        darkMode = sharedPref.getBoolean("darkMode", false)
        duplicateVideos = sharedPref.getBoolean("duplicateVideos", true)
        suggestionMode = sharedPref.getString("suggestionMode", "nearest") ?: "nearest"
    }

    private fun setupUI() {
        val sharedPref = getSharedPreferences("AppSettings", MODE_PRIVATE)

        switchDuplicateVideos.isChecked = duplicateVideos
        switchDarkMode.isChecked = darkMode

        when (suggestionMode) {
            "llm" -> radioLLM.isChecked = true
            "nearest" -> radioNearestNeighbor.isChecked = true
            "none" -> radioNone.isChecked = true
        }

        switchDuplicateVideos.setOnCheckedChangeListener { _, isChecked ->
            sharedPref.edit().putBoolean("duplicateVideos", isChecked).apply()
        }

        switchDarkMode.setOnCheckedChangeListener { _, isChecked ->
            sharedPref.edit().putBoolean("darkMode", isChecked).apply()
        }

        suggestionsRadioGroup.setOnCheckedChangeListener { _, checkedId ->
            val selectedMode = when (checkedId) {
                R.id.radioLLM -> "llm"
                R.id.radioNearestNeighbor -> "nearest"
                R.id.radioNone -> "none"
                else -> "nearest"
            }
            sharedPref.edit().putString("suggestionMode", selectedMode).apply()
            Log.d("RADIO", "Selected suggestion mode: $selectedMode")
        }
    }


    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            android.R.id.home -> {
                val resultIntent = Intent().apply {
                    putExtra("allowDuplicateVideos", duplicateVideos)
                    putExtra("darkMode", darkMode)
                    putExtra("suggestionMode", suggestionMode)
                }
                setResult(RESULT_OK, resultIntent)
                finish()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }
}
