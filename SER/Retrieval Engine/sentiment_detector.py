import os
import cv2
import asyncio
import matplotlib.pyplot as plt
from deepface import DeepFace
from transformers import pipeline
from collections import defaultdict
import whisper
from PIL import Image
from moviepy import VideoFileClip

class SentimentDetector:
    def __init__(self):
        # models
        #self.pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
        self.pipe2 = pipeline("image-classification", model="trpakov/vit-face-expression")
        self.emotion_text_classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")
        self.whisper = whisper.load_model("base").half().to("cuda")

    @staticmethod
    def convert_mp4_to_mp3(mp4_file, mp3_file, start_time=0, end_time=1000):
        if not os.path.exists(mp4_file):
            raise FileNotFoundError(f"Error: The file {mp4_file} was not found.")

        video = VideoFileClip(mp4_file)

        if end_time is None or end_time > video.duration:
            end_time = video.duration

        audio = video.audio.subclipped(start_time, end_time)
        audio.write_audiofile(mp3_file)
        audio.close()
        video.close()

    @staticmethod
    def get_text_from_mp3(audio_file):
        model = whisper.load_model("tiny")  # "tiny" or "small" to avoid memory issues, change on node
        result = model.transcribe(audio_file)

        if "text" in result:
            return result["text"]
        else:
            return None

    def get_emotion_from_text(self, text:str):
        emotion = self.emotion_text_classifier(text) # output: [{'label': 'joy', 'score': 0.9887555241584778}]
        return emotion

    """
    async def get_emotion_for_image(self, image):
        predictions = self.pipe(image)
        top_emotion = predictions[0]["label"]
        confidence = predictions[0]["score"]
        return top_emotion, confidence"""

    def get_emotion_for_image2(self, image):
        predictions = self.pipe2(image)
        top_emotion = predictions[0]["label"]
        confidence = predictions[0]["score"]
        return top_emotion, confidence

    def detect_faces_and_get_emotion_with_plots(self, file_path):
        if not os.path.exists(file_path):
            print("File path doesn't exist")
            return None, None, None, None

        faces_dir = "./faces"
        os.makedirs(faces_dir, exist_ok=True)  # Ensure the faces directory exists

        img = cv2.imread(file_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = DeepFace.extract_faces(file_path, detector_backend="retinaface", enforce_detection=False)

        if not faces:
            print("No faces detected")
            return None, None, None, None

        emotion_scores = defaultdict(float)
        total_confidence = 0
        total_faces = 0

        for i, face_data in enumerate(faces):
            x1, y1 = face_data["facial_area"]["x"], face_data["facial_area"]["y"]
            width, height = face_data["facial_area"]["w"], face_data["facial_area"]["h"]
            x2, y2 = x1 + width, y1 + height
            face = img_rgb[y1:y2, x1:x2]
            face_pil = Image.fromarray(face)

            emotion, confidence = asyncio.run(self.get_emotion_for_image2(face_pil))
            emotion_scores[emotion] += confidence
            total_confidence += confidence
            total_faces += 1

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{emotion} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)

        # Save the annotated image
        annotated_path = os.path.join(faces_dir, f"annotated_{os.path.basename(file_path)}")
        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.savefig(annotated_path)
        plt.close()  # Close figure to free memory

        print(f"Annotated face image saved at: {annotated_path}")

        if total_faces == 0 or total_confidence == 0:
            return None, None, None, None

        weighted_avg_emotion = {emotion: score / total_confidence for emotion, score in emotion_scores.items()}
        dominant_emotion = max(weighted_avg_emotion, key=weighted_avg_emotion.get)
        dominant_sentiment = self.get_sentiment_from_emotion(dominant_emotion)
        return dominant_emotion, weighted_avg_emotion[dominant_emotion], dominant_sentiment, annotated_path


    def detect_faces_and_get_emotion(self, file_path):
        if not os.path.exists(file_path):
            print("File path doesn't exist")
            return

        img = cv2.imread(file_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = DeepFace.extract_faces(file_path, detector_backend="retinaface", enforce_detection=False)

        if not faces:
            print("No faces detected")
            return

        sentiment_scores = defaultdict(float)
        total_confidence = 0
        total_faces = 0

        for i, face_data in enumerate(faces):
            x1, y1 = face_data["facial_area"]["x"], face_data["facial_area"]["y"]
            width, height = face_data["facial_area"]["w"], face_data["facial_area"]["h"]
            x2, y2 = x1 + width, y1 + height
            face = img_rgb[y1:y2, x1:x2]
            face_pil = Image.fromarray(face)

            emotion, confidence = asyncio.run(self.get_emotion_for_image2(face_pil))

            # Store weighted sentiment calculation per image
            sentiment_scores[emotion] += confidence
            total_confidence += confidence
            total_faces += 1

        if total_faces == 0:
            print("no faces detected in this image")
            return

        weighted_avg_sentiment = {emotion: score / total_confidence for emotion, score in sentiment_scores.items()}
        return weighted_avg_sentiment


    @staticmethod
    def get_sentiment_from_emotion(emotion:str):
        emotion_to_sentiment = {
            "happy": "positive",
            "sad": "negative",
            "surprise": "positive",
            "angry": "negative",
            "neutral": "neutral",
            "disgust": "negative",
            "fear": "negative"
        }
        return emotion_to_sentiment.get(emotion.lower(), "neutral")

"""
async def main():
    SD = SentimentDetector()

    # SD.convert_mp4_to_mp3("./videos/00002.mp4", "./mp3/00002.mp3")
    text = await SD.get_text_from_mp3("./mp3/00002.mp3")
    print(text)
    sentiment = await SD.get_emotion_from_text(text)
    print(sentiment)
SD = SentimentDetector()
results1 = SD.detect_faces_and_get_emotion_with_plots("./frames/00001.mp4_frame_6307.jpg")
print("Image 1 Average Sentiment:", results1)

results2 = SD.detect_faces_and_get_emotion_with_plots("./frames/00001.mp4_frame_4350.jpg")
print("Image 2 Average Sentiment:", results2)

results3 = SD.detect_faces_and_get_emotion_with_plots("./frames/00003.mp4_frame_8922.jpg")
print("Image 3 Average Sentiment:", results3)

results4 = SD.detect_faces_and_get_emotion_with_plots("./frames/00003.mp4_frame_10809.jpg")
print("Image 4 Average Sentiment:", results4)

results5 = SD.detect_faces_and_get_emotion_with_plots("./frames/00002.mp4_frame_2888.jpg")
print("Image 5 Average Sentiment:", results5)
#asyncio.run(main())"""