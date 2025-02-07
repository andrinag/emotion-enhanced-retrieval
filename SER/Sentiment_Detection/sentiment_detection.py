from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import pipeline

pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")


class SentimentDetector:

    @staticmethod
    def get_sentiment(self, image_path: str):
        # image_path = "faces/surprise/2Q___face.png" # static image
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
        return top_emotion
