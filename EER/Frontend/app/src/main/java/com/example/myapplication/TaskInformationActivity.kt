package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import kotlinx.coroutines.*
import org.openapitools.client.apis.EvaluationClientApi

data class EvaluationDisplay(
    val id: String,
    val name: String,
    val taskName: String?,
    val taskType: String?
)

class TaskInformationActivity : AppCompatActivity() {

    private lateinit var doneButton: Button
    private lateinit var recyclerView: RecyclerView
    private var selectedEvaluationId: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_task_information)

        doneButton = findViewById(R.id.buttonDone)
        recyclerView = findViewById(R.id.evaluationRecyclerView)
        recyclerView.layoutManager = LinearLayoutManager(this)

        doneButton.setOnClickListener {
            if (selectedEvaluationId == null) {
                Toast.makeText(this, "Please select an evaluation", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            getSharedPreferences("UserSettings", MODE_PRIVATE)
                .edit()
                .putString("selectedEvaluationId", selectedEvaluationId)
                .apply()

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

                if (evaluations.isEmpty()) {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@TaskInformationActivity, "No evaluations found.", Toast.LENGTH_LONG).show()
                    }
                    return@launch
                }

                val displayList = mutableListOf<EvaluationDisplay>()
                for (eval in evaluations) {
                    val evalId = eval.id ?: continue
                    val evalName = eval.name ?: continue
                    val task = try {
                        api.getApiV2ClientEvaluationCurrentTaskByEvaluationId(evalId, session)
                    } catch (e: Exception) {
                        continue
                    }
                    displayList.add(
                        EvaluationDisplay(
                            id = evalId,
                            name = evalName,
                            taskName = task?.name,
                            taskType = task?.taskType
                        )
                    )
                }

                withContext(Dispatchers.Main) {
                    recyclerView.adapter = EvaluationAdapter(displayList) { selected ->
                        selectedEvaluationId = selected.id
                        prefs.edit().putString("evaluationId", selectedEvaluationId).apply()
                    }
                }

            } catch (e: Exception) {
                Log.e("TASK_INFO", "Failed to load evaluations: ${e.message}")
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@TaskInformationActivity, "Error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }
}
