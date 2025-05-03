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
    private lateinit var switchCheerupMode: Switch
    private lateinit var jokesCheckbox: CheckBox
    private lateinit var complimentsCheckbox: CheckBox
    private lateinit var suggestionsRadioGroup: RadioGroup
    private lateinit var radioLLM: RadioButton
    private lateinit var radioNearestNeighbor: RadioButton
    private lateinit var radioNone: RadioButton

    private var duplicateVideos = true
    private var darkMode = true
    private var cheerupMode = false
    private var jokesActivated = false
    private var complimentsActivated = false
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
        switchCheerupMode = findViewById(R.id.cheerupMode)
        jokesCheckbox = findViewById(R.id.jokeCheckbox)
        complimentsCheckbox = findViewById(R.id.complimentCheckbox)
        suggestionsRadioGroup = findViewById(R.id.suggestionsRadioGroup)
        radioLLM = findViewById(R.id.radioLLM)
        radioNearestNeighbor = findViewById(R.id.radioNearestNeighbor)
        radioNone = findViewById(R.id.radioNone)
    }

    private fun loadPreferences() {
        val sharedPref = getSharedPreferences("AppSettings", MODE_PRIVATE)
        darkMode = sharedPref.getBoolean("darkMode", false)
        duplicateVideos = sharedPref.getBoolean("duplicateVideos", true)
        cheerupMode = sharedPref.getBoolean("cheerupMode", false)
        jokesActivated = sharedPref.getBoolean("jokesActivated", false)
        complimentsActivated = sharedPref.getBoolean("complimentsActivated", false)
        suggestionMode = sharedPref.getString("suggestionMode", "nearest") ?: "nearest"
    }

    private fun setupUI() {
        val sharedPref = getSharedPreferences("AppSettings", MODE_PRIVATE)

        switchDuplicateVideos.isChecked = duplicateVideos
        switchDarkMode.isChecked = darkMode
        switchCheerupMode.isChecked = cheerupMode
        jokesCheckbox.isChecked = jokesActivated
        complimentsCheckbox.isChecked = complimentsActivated

        when (suggestionMode) {
            "llm" -> radioLLM.isChecked = true
            "nearest" -> radioNearestNeighbor.isChecked = true
            "none" -> radioNone.isChecked = true
        }

        val cheerupOptionsLayout = findViewById<LinearLayout>(R.id.cheerupOptionsLayout)
        cheerupOptionsLayout.visibility = if (cheerupMode) View.VISIBLE else View.GONE

        switchDuplicateVideos.setOnCheckedChangeListener { _, isChecked ->
            sharedPref.edit().putBoolean("duplicateVideos", isChecked).apply()
        }

        switchDarkMode.setOnCheckedChangeListener { _, isChecked ->
            sharedPref.edit().putBoolean("darkMode", isChecked).apply()
        }

        switchCheerupMode.setOnCheckedChangeListener { _, isChecked ->
            cheerupOptionsLayout.visibility = if (isChecked) View.VISIBLE else View.GONE
            sharedPref.edit().putBoolean("cheerupMode", isChecked).apply()
        }

        jokesCheckbox.setOnCheckedChangeListener { _, isChecked ->
            sharedPref.edit().putBoolean("jokesActivated", isChecked).apply()
            validateCheerupOptions(sharedPref)
        }

        complimentsCheckbox.setOnCheckedChangeListener { _, isChecked ->
            sharedPref.edit().putBoolean("complimentsActivated", isChecked).apply()
            validateCheerupOptions(sharedPref)
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

    private fun validateCheerupOptions(sharedPref: android.content.SharedPreferences) {
        if (!jokesCheckbox.isChecked && !complimentsCheckbox.isChecked) {
            jokesCheckbox.isChecked = true
            sharedPref.edit().putBoolean("jokesActivated", true).apply()
            Toast.makeText(this, "At least one option must be selected.", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            android.R.id.home -> {
                val resultIntent = Intent().apply {
                    putExtra("allowDuplicateVideos", duplicateVideos)
                    putExtra("darkMode", darkMode)
                    putExtra("cheerupMode", cheerupMode)
                    putExtra("jokesActivated", jokesActivated)
                    putExtra("complimentsActivated", complimentsActivated)
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
