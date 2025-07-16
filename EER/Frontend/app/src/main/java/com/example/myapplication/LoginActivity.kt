package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.openapitools.client.apis.UserApi
import org.openapitools.client.models.LoginRequest


class LoginActivity : AppCompatActivity() {
    private lateinit var loginButton: Button
    private lateinit var usernameField: EditText
    private lateinit var passwordField: EditText
    private lateinit var userlessButton: Button
    private var userApi = UserApi("http://10.34.64.205:8080")


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_login)

        loginButton = findViewById(R.id.loginButton)
        userlessButton = findViewById(R.id.userlessButton)
        usernameField = findViewById(R.id.usernameField)
        passwordField = findViewById(R.id.passwordField)


        loginButton.setOnClickListener {
            val username = usernameField.text.toString().trim()
            val password = passwordField.text.toString().trim()

            if (username.isEmpty() || password.isEmpty()) {
                Toast.makeText(this, "Please enter both username and password", Toast.LENGTH_SHORT).show()
            } else {
                loginUser(username, password)
            }
        }

        userlessButton.setOnClickListener {
            val userPref = getSharedPreferences("UserSettings", MODE_PRIVATE)
            userPref.edit().putString("username", null).apply()
            val intent = Intent(this@LoginActivity, MainActivity::class.java)
            startActivity(intent)
        }
    }


    private fun loginUser(username: String, password: String) {
        val loginRequest = LoginRequest(
            username = username,
            password = password
        )
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val user = userApi.postApiV2Login(loginRequest)
                Log.d("LOGIN", "Reponse ${user.sessionId}")
                Log.d("LOGIN", "User logged in: ${user.username}")  // replace with actual field
                launch(Dispatchers.Main) {
                    val userPref = getSharedPreferences("UserSettings", MODE_PRIVATE)
                    userPref.edit().putString("username", user.username).apply()
                    userPref.edit().putString("sessionId", user.sessionId).apply()
                    Toast.makeText(this@LoginActivity, "Welcome, ${user.username}!", Toast.LENGTH_SHORT).show()
                    val intent = Intent(this@LoginActivity, TaskInformationActivity::class.java)
                    startActivity(intent)
                    finish()
                }

            } catch (e: Exception) {
                Log.e("LOGIN", "Login failed: ${e.message}")
                launch(Dispatchers.Main) {
                    Toast.makeText(this@LoginActivity, "Login failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

}