from PIL.Image import Image
from fastapi import FastAPI
import uvicorn
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from pydantic import BaseModel
import base64
from PIL import Image
import io



app = FastAPI()

pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")

class ImageRequest(BaseModel):
    image: str  # Base64 encoded image


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
    for i in range(0, len(predictions)):
        top_emotion = predictions[i]["label"]
        confidence = predictions[i]["score"]
        print(f"Predicted Emotion: {top_emotion} ({confidence:.2f})")

    return top_emotion, sentiment


@app.api_route("/test", methods=["GET", "POST"])
async def test():
    return "hello"

@app.post("/upload_base64")
async def upload_base64_image(data: ImageRequest):
    try:
        image_data = base64.b64decode(data.image)
        image = Image.open(io.BytesIO(image_data))
        image.save("output_image.png", "PNG")
        top_emotion, sentiment = await get_sentiment_for_image("output_image.png")

        return {"sentiment": sentiment, "emotion": top_emotion}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
