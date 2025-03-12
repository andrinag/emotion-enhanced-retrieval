from PIL.Image import Image
from fastapi import FastAPI
import uvicorn
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from pydantic import BaseModel
import base64
from PIL import Image
import io
import cv2



app = FastAPI()

############################# SENTIMENT OF FACE #####################################

class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

pipe_sentiment_face = pipeline("image-classification", model="trpakov/vit-face-expression")


async def get_sentiment_for_image(image_path: str):
    predictions = pipe_sentiment_face(image_path)
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
    """for i in range(0, len(predictions)):
        emotion = predictions[i]["label"]
        confidence1 = predictions[i]["score"]
        print(f"Predicted Emotion: {emotion} ({confidence1:.2f})")"""

    return top_emotion, sentiment

@app.post("/upload_base64")
async def upload_base64_image(data: ImageRequest):
    try:
        image_data = base64.b64decode(data.image)
        image = Image.open(io.BytesIO(image_data))
        image = image.rotate(90.0)
        image.save("output_image.png", "PNG")
        top_emotion, sentiment = await get_sentiment_for_image("output_image.png")

        return {"sentiment": sentiment, "emotion": top_emotion}
    except Exception as e:
        return {"error": str(e)}


######################## SENTIMENT OF TEXT QUERY ###################################


class QueryRequest(BaseModel):
    query: str

sentiment_text_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    return_all_scores=True
)
emotion_text_classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")

async def get_sentiment_for_query(query: str):
    sentiment = sentiment_text_classifier(query)
    return sentiment


async def get_emotion_for_query(query: str):
    emotion = emotion_text_classifier(query)
    return emotion

@app.post("/upload_query")
async def upload_query(data: QueryRequest):
    try:
        emotion = await get_emotion_for_query(data.query)
        return {"emotion": emotion}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
