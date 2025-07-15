package com.example.myapplication

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.openapitools.client.apis.UserApi
import org.openapitools.client.models.LoginRequest


class LoginActivity : AppCompatActivity() {
    private lateinit var loginButton: Button
    private lateinit var usernameField: EditText
    private lateinit var passwordField: EditText
    private var userApi = UserApi("http://10.34.64.205:8080")


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_login)

        loginButton = findViewById(R.id.loginButton)
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
    }


    private fun loginUser(username: String, password: String) {
        val loginRequest = LoginRequest(
            username = "andrina",
            password = "emotion"
        )
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val user = userApi.postApiV2Login(loginRequest)
                Log.d("LOGIN", "User logged in: ${user.username}")  // or whatever field the response has

            } catch (e: Exception) {
                Log.e("LOGIN", "Login failed: ${e.message}")
            }
        }



    }
}