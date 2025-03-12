import os
import cv2
import torch
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
from PIL import Image
from transformers import pipeline
from collections import defaultdict


class SentimentDetector:
    def __init__(self):
        # sentiment models
        self.pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
        self.pipe2 = pipeline("image-classification", model="trpakov/vit-face-expression")

    async def get_sentiment_for_image(self, image):
        predictions = self.pipe(image)
        top_emotion = predictions[0]["label"]
        confidence = predictions[0]["score"]
        return top_emotion, confidence

    async def get_sentiment_for_image2(self, image):
        predictions = self.pipe2(image)
        top_emotion = predictions[0]["label"]
        confidence = predictions[0]["score"]
        return top_emotion, confidence

    def detect_faces_and_get_sentiment(self, file_path):
        if not os.path.exists(file_path):
            print("File path doesn't exist")
            return

        img = cv2.imread(file_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = DeepFace.extract_faces(file_path, detector_backend="retinaface")

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

            emotion, confidence = asyncio.run(self.get_sentiment_for_image2(face_pil))

            # Store weighted sentiment calculation per image
            sentiment_scores[emotion] += confidence
            total_confidence += confidence
            total_faces += 1

            # Draw rectangle and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{emotion} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)

        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

        if total_faces == 0:
            return "No faces detected in this image."

        weighted_avg_sentiment = {emotion: score / total_confidence for emotion, score in sentiment_scores.items()}
        return weighted_avg_sentiment


if __name__ == "__main__":
    SD = SentimentDetector()
    results1 = SD.detect_faces_and_get_sentiment("./frames/00001.mp4_frame_6307.jpg")
    print("Image 1 Average Sentiment:", results1)

    results2 = SD.detect_faces_and_get_sentiment("./frames/00001.mp4_frame_4350.jpg")
    print("Image 2 Average Sentiment:", results2)

    results3 = SD.detect_faces_and_get_sentiment("./frames/00003.mp4_frame_8922.jpg")
    print("Image 3 Average Sentiment:", results3)

    results4 = SD.detect_faces_and_get_sentiment("./frames/00003.mp4_frame_10809.jpg")
    print("Image 4 Average Sentiment:", results4)

    results5 = SD.detect_faces_and_get_sentiment("./frames/00002.mp4_frame_2888.jpg")
    print("Image 5 Average Sentiment:", results5)