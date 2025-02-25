package com.example.myapplication
import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import android.widget.Button


class MainActivity : AppCompatActivity() {


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // starts sentiment detection when button is clicked
        val startSentimentButton = findViewById<Button>(R.id.startSentimentButton)
        startSentimentButton.setOnClickListener {
            val intent = Intent(this, SentimentActivity::class.java)
            startActivity(intent)
        }

        // starts video stream when clicked
        val startVideosButton = findViewById<Button>(R.id.startVideosButton)
        startVideosButton.setOnClickListener {
            val intent = Intent(this, VideoActivity::class.java)
            startActivity(intent)
        }
    }
}
