package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.MenuItem
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import kotlin.apply


/**
 * Activity for the settings where user can handle their preferences.
 * Is activated whenever a user clicks on the settings button in the MainActivity.
 */
class SettingsActivity : AppCompatActivity() {

    private lateinit var switchDuplicateVideos: Switch
    private lateinit var switchDarkMode: Switch
    private lateinit var suggestionsRadioGroup: RadioGroup
    private lateinit var radioLLM: RadioButton
    private lateinit var radioNearestNeighbor: RadioButton
    private lateinit var radioNone: RadioButton
    private lateinit var switchCheerupMode: Switch
    private lateinit var jokesCheckbox : CheckBox
    private lateinit var complimentsCheckbox : CheckBox
    private lateinit var emotionSwitch: Switch
    private var cheerupMode = false;
    private var jokesActivated = false;
    private var complimentsActivated = false;
    private var duplicateVideos = true
    private var darkMode = true
    private var suggestionMode: String = "nearest"
    private var emotionMode = true

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
        switchCheerupMode = findViewById(R.id.cheerupMode)
        jokesCheckbox = findViewById(R.id.jokeCheckbox)
        complimentsCheckbox = findViewById(R.id.complimentCheckbox)
        emotionSwitch = findViewById(R.id.emotionCheckbox)
    }

    /**
     * Loads the preferences of the user that are saved in a sharedPreferences datatype.
     */
    private fun loadPreferences() {
        val sharedPref = getSharedPreferences("AppSettings", MODE_PRIVATE)
        darkMode = sharedPref.getBoolean("darkMode", false)
        duplicateVideos = sharedPref.getBoolean("duplicateVideos", true)
        suggestionMode = sharedPref.getString("suggestionMode", "nearest") ?: "nearest"
        emotionMode = sharedPref.getBoolean("emotionMode", false)
    }

    /**
     * Sets up all of the buttons and sets them to the current preferences of the user.
     */
    private fun setupUI() {
        val sharedPref = getSharedPreferences("AppSettings", MODE_PRIVATE)

        switchDuplicateVideos.isChecked = duplicateVideos
        switchDarkMode.isChecked = darkMode
        emotionSwitch.isChecked = emotionMode

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
        emotionSwitch.setOnCheckedChangeListener { _, isChecked ->
            sharedPref.edit().putBoolean("emotionMode", isChecked).apply()

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

        cheerupMode = sharedPref.getBoolean("cheerupMode", false)
        jokesActivated = sharedPref.getBoolean("jokesActivated", false)
        complimentsActivated = sharedPref.getBoolean("complimentsActivated", false)
        switchCheerupMode.isChecked = cheerupMode
        jokesCheckbox.isChecked = jokesActivated
        complimentsCheckbox.isChecked = complimentsActivated
        emotionMode = sharedPref.getBoolean("emotionMode", false)

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
                val resultIntent = Intent().apply {
                    putExtra("allowDuplicateVideos", duplicateVideos)
                    putExtra("darkMode", darkMode)
                    putExtra("suggestionMode", suggestionMode)
                    putExtra("emotionMode", emotionMode)
                }
                setResult(RESULT_OK, resultIntent)
                finish()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }
}
