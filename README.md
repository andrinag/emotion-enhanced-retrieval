# Emotion-Enhanced Multimedia Retrieval Engine

This repository contains the code for the bachelor's thesis **Enhancing Multimedia Retrieval with Emotion-Aware Augmented Reality Interfaces**. It includes:

- A retrieval engine that supports emotion and sentiment-aware search
- An Android application used as the user interface

---

## User Setup (Android Application)

1. Ensure you have [Android Studio](https://developer.android.com/studio) installed.
2. Clone this repository locally.
3. Put your Android device (Android version 8.1 / API 27 or higher) into **Developer Mode**.
4. Enable **USB Debugging** and connect your device to your computer via USB.
5. Open the project in Android Studio and run the app. It should automatically install and launch on your device.
6. To connect to the university’s internal services, install the **GlobalProtect VPN** app and log in using your university credentials.

---

## Developer Setup

### Data Insertion

1. Ensure **Python 3.10 or higher** is installed.
2. Install the required Python packages:
   ```bash
   pip install -r Engine/requirements.txt
   ```
3. A system with a GPU is recommended, as multiple machine learning models are used.
4. Create a database called multimedia_db (or rename it in the code) and create the schema according to DB_schema.sql.

#### Video Insertion

1. Start the backend insertion engine (default port is `8000`):
   ```bash
   python3 emotion_enhanced_engine.py
   ```
2. Run the script to insert videos into the database:
   ```bash
   python3 upload_scripts/upload_videos.py
   ```

This process will insert all the videos and extract semantical and emotional features. Be aware that this process might take a long time. 

4. After the process is complete, run the OCR insertion script: 
   ```bash
   python3 insert_OCR_data.py
   ```

5. Create the index on the embeddings:
   ```bash
   python3 create_index_embedding.py
   ```
When all of the videos and OCR data are inserted, please start the rest of the backend processes, such that the frontend can run as intended. 

### Backend

1. Start the search API:
```bash
   python3 search_api.py
   ```
2. Start the sentiment API :
```bash
   python3 Sentiment_Detection/sentiment_api.py
   ```
3. Start the tinyLlama model on the server. This works best when run with Ollama. 

## Frontend (Android App)

Same steps as under **User Setup**. Make sure to:

- Enable USB Debugging
- Connect to the GlobalProtect VPN for the university on the mobile device. 

Run the project from Android Studio, and the app should deploy to your connected device.

---

## Models Used

This project uses the following models:

- **Text-to-Emotion**: [michellejieli/emotion\_text\_classifier](https://huggingface.co/michellejieli/emotion_text_classifier)
- **Facial Expression Recognition**: [trpakov/vit-face-expression](https://huggingface.co/trpakov/vit-face-expression)
- **Face Recognition**: [DeepFace](https://github.com/serengil/deepface)
- **Speech-to-Text**: [Whisper](https://github.com/openai/whisper)
- **Encoding Images to Embeddings**: [CLIP](https://github.com/openai/CLIP)
- **TinyLlama**: [TinyLlama](https://ollama.com/library/tinyllama)


## Other frameworks 
- **HTTP Requests Android**: [Volley](https://google.github.io/volley/)
