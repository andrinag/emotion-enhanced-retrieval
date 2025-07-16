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
    private lateinit var doneButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_task_information)

        taskInfoText = findViewById(R.id.taskInfoText)
        doneButton = findViewById(R.id.buttonDone)

        doneButton.setOnClickListener {
            startActivity(Intent(this, MainActivity::class.java))
            finish()
        }

        fetchEvaluationsAndTasks()
    }

    private fun fetchEvaluationsAndTasks() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val prefs = getSharedPreferences("UserSettings", MODE_PRIVATE)
                val session = prefs.getString("sessionId", null)
                val api = EvaluationClientApi("http://10.34.64.205:8080")
                val evaluations = api.getApiV2ClientEvaluationList(session)
                val evaluations2 = api.getApiV2ClientEvaluationListWithHttpInfo(session)
                Log.d("EVAL", "evaluations2: ${evaluations2}")

                if (evaluations.isEmpty()) {
                    withContext(Dispatchers.Main) {
                        taskInfoText.text = "No evaluations found."
                    }
                    return@launch
                }

                val sb = StringBuilder()
                sb.append("Evaluations and Current Tasks:\n\n")

                for (eval in evaluations) {
                    val evalId = eval.id ?: continue
                    val evalName = eval.name ?: continue
                    sb.append("Evaluation ID: $evalId\n")
                    sb.append("Evaluation name: $evalName\n")

                    val task = try {
                        api.getApiV2ClientEvaluationCurrentTaskByEvaluationId(evalId, session)
                    } catch (e: Exception) {
                        Log.w("TASK_FETCH", "No task for evaluation $evalId: ${e.message}")
                        null
                    }

                    if (task != null) {
                        sb.append("• Task Name: ${task.name}\n")
                        sb.append("• Task Status: ${task.taskType}\n\n")
                    } else {
                        sb.append("• No active task.\n\n")
                    }
                }

                withContext(Dispatchers.Main) {
                    taskInfoText.text = sb.toString()
                }

            } catch (e: Exception) {
                Log.e("TASK_INFO", "Failed to load evaluations: ${e.message}")
                withContext(Dispatchers.Main) {
                    taskInfoText.text = "Failed to load task information.\n${e.message}"
                }
            }
        }
    }
}
