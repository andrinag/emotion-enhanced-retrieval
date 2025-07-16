package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.*
import org.openapitools.client.apis.EvaluationClientApi

class TaskInformationActivity : AppCompatActivity() {

    private lateinit var taskInfoText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_task_information)

        taskInfoText = findViewById(R.id.taskInfoText)

        fetchEvaluationsAndTasks()

        findViewById<Button>(R.id.buttonDone).setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
            finish()
        }

    }

    private fun fetchEvaluationsAndTasks() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val userPref = getSharedPreferences("UserSettings", MODE_PRIVATE)
                val username = userPref.getString("username", null)
                val sessionId = userPref.getString("sessionId", null)
                val api = EvaluationClientApi("http://10.34.64.205:8080")
                val evaluations = api.getApiV2ClientEvaluationList(session = sessionId)

                val sb = StringBuilder()
                sb.append("Evaluations & Current Tasks:\n\n")

                for (evaluation in evaluations) {
                    val evalId = evaluation.id ?: continue
                    val task = try {
                        api.getApiV2ClientEvaluationCurrentTaskByEvaluationId(evalId, session = sessionId)
                    } catch (e: Exception) {
                        null
                    }

                    sb.append("Evaluation ID: $evalId\n")
                    sb.append("→ Task: ${task?.name ?: "No current task"}\n\n")
                }

                withContext(Dispatchers.Main) {
                    taskInfoText.text = sb.toString()
                }

            } catch (e: Exception) {
                Log.e("TASK_INFO", "Failed to load evaluations: ${e.message}")
                withContext(Dispatchers.Main) {
                    taskInfoText.text = "Failed to load task information."
                }
            }
        }
    }
}
