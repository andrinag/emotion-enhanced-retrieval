import psycopg2
from fastapi import FastAPI, UploadFile, File, Response, Header, Request, HTTPException
import uvicorn
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification

app = FastAPI()

pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")


@app.post("/sentiment_for_image/{image_path}")
async def get_sentiment_for_image(image_path: str):
    predictions = pipe(image_path)
    top_emotion = predictions[0]["label"]
    confidence = predictions[0]["score"]

    emotion_to_sentiment = {
        "angry": "negative",
        "disgust": "negative",
        "fear": "negative",
        "sad": "negative",
        "neutral": "neutral",
        "happy": "positive",
        "surprise": "positive"
    }

    sentiment = emotion_to_sentiment.get(top_emotion, "unknown")

    print(f"Predicted Emotion: {top_emotion} ({confidence:.2f})")
    print(f"Mapped Sentiment: {sentiment}")

    return sentiment


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
