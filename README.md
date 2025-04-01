# Emotion-Enhanced Multimedia Retrieval Engine

This repository contains the code for the bachelorthesis **Enhancing Multimedia Retrieval with Emotion-Aware Augmented Reality Interfaces**. It includes:

- A retrieval engine that supports emotion and sentiment-aware search
- An Android application used as the user interface
- A sentiment API that processes user expressions and queries in a sentiment and emotion enhanced manner

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

### Backend

1. Ensure **Python 3.10 or higher** is installed.
2. Install the required Python packages:
   ```bash
   pip install -r Engine/requirements.txt
   ```
3. A system with a GPU is recommended, as multiple machine learning models are used.

#### Video Insertion

1. Start the backend engine (default port is `8000`):
   ```bash
   python3 engine.py
   ```
2. Run the script to insert videos into the database:
   ```bash
   python3 upload_videos.py
   ```

This will:

- Generate embeddings for video frames
- Analyze sentiment from facial expressions, OCR text, and audio (ASR)
- Store all extracted data and metadata in the database

Depending on the number of videos, this process can take some time.

#### API Usage

Once videos are inserted, start the retrieval API:

```bash
python api.py
```

This API handles search requests that include query text and optional emotion/sentiment constraints and returns multimedia content matching the request.

---

## Frontend (Android App)

Same steps as under **User Setup**. Make sure to:

- Enable USB Debugging
- Connect to GlobalProtect VPN if required

Run the project from Android Studio, and the app should deploy to your connected device.

---

## Models Used

This project uses the following models:

- **Text-to-Emotion**: [michellejieli/emotion\_text\_classifier](https://huggingface.co/michellejieli/emotion_text_classifier)
- **Facial Expression Recognition**: [trpakov/vit-face-expression](https://huggingface.co/trpakov/vit-face-expression)
- **Face Recognition**: [DeepFace](https://github.com/serengil/deepface)
- **Speech-to-Text**: [Whisper](https://github.com/openai/whisper)
- **Encoding Images to Embeddings**: [CLIP](https://github.com/openai/CLIP)


## Other frameworks 
- **HTTP Requests Android**: [Volley](https://google.github.io/volley/)
