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
from transformers import *
import librosa
import torch
from pydub import AudioSegment

"""
Sentiment Detector Class can be created as an object. The classification for emotions during the 
data insertion process is handled through that object. 
"""
class SentimentDetector:
    def __init__(self):
        # models
        self.pipe2 = pipeline("image-classification", model="trpakov/vit-face-expression")
        self.emotion_text_classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")
        self.whisper = whisper.load_model("base")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")
        self.model = Wav2Vec2ForCTC.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")

    @staticmethod
    def convert_mp4_to_mp3(mp4_file, mp3_file, start_time=0, end_time=1000):
        """
        Converts a video file into an audio file. Start and entime are given.
        """
        if not os.path.exists(mp4_file):
            raise FileNotFoundError(f"Error: The file {mp4_file} was not found.")

        video = VideoFileClip(mp4_file)

        if end_time is None or end_time > video.duration:
            end_time = video.duration

        audio = video.audio.subclipped(start_time, end_time)
        audio.write_audiofile(mp3_file)
        audio.close()
        video.close()


    def predict_emotion_speech_acoustic(self, audio_path):
        try:
            sound = AudioSegment.from_mp3(audio_path)
            sound.export("audio.wav", format="wav")
        except Exception as e:
            print(f"Could not convert mp3 to wav: {e}")

        try:
            audio, rate = librosa.load("audio.wav", sr=16000)
            inputs = self.feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)

            with torch.no_grad():
                outputs = self.model(inputs.input_values)
                predictions = torch.nn.functional.softmax(outputs.logits.mean(dim=1),
                                                          dim=-1)  # Average over sequence length
                predicted_label = torch.argmax(predictions, dim=-1)
                emotion = self.model.config.id2label[predicted_label.item(), "unknown"]
            return emotion
        except Exception as e:
            print("Could not get emotion from wav")

    @staticmethod
    def get_text_from_mp3(audio_file):
        """
        Performs the speech to text conversion.
        """
        model = whisper.load_model("tiny")  # "tiny" or "small" to avoid memory issues, change on node
        result = model.transcribe(audio_file)

        if "text" in result:
            return result["text"]
        else:
            return None

    def get_emotion_from_text(self, text:str):
        """Return emotion classification for texts. """
        emotion = self.emotion_text_classifier(text) # output: [{'label': 'joy', 'score': 0.9887555241584778}]
        return emotion


    def get_emotion_for_image(self, image):
        """Facial emotion classification that returns only the top emotion and corresponding confidence. """
        predictions = self.pipe2(image)
        top_emotion = predictions[0]["label"]
        confidence = predictions[0]["score"]
        return top_emotion, confidence

    def detect_faces_and_get_emotion_with_plots(self, file_path):
        """
        Detects all the faces in the images with DeepFace, gets their classified emotion and draws green squares around
        the faces with emotion and confidence.
        """
        if not os.path.exists(file_path):
            print("File path doesn't exist")
            return "no_face", 0.0, "neutral", None

        faces_dir = "./faces"
        os.makedirs(faces_dir, exist_ok=True)

        img = cv2.imread(file_path)
        if img is None:
            print(f"[ERROR] Could not read image: {file_path}")
            return "no_face", 0.0, "neutral", None

        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[ERROR] cvtColor failed: {e}")
            return "no_face", 0.0, "neutral", None

        try:
            faces = DeepFace.extract_faces(file_path, detector_backend="retinaface", enforce_detection=True)
        except Exception as e:
            print(f"[ERROR] DeepFace failed on {file_path}: {e}")
            return "no_face", 0.0, "neutral", None

        if not faces:
            print("No faces detected")
            return "no_face", 0.0, "neutral", None

        emotion_scores = defaultdict(float)
        total_confidence = 0
        total_faces = 0

        for i, face_data in enumerate(faces):
            x1, y1 = face_data["facial_area"]["x"], face_data["facial_area"]["y"]
            width, height = face_data["facial_area"]["w"], face_data["facial_area"]["h"]
            x2, y2 = x1 + width, y1 + height
            face = img_rgb[y1:y2, x1:x2]
            face_pil = Image.fromarray(face)

            emotion, confidence = self.get_emotion_for_image(face_pil)
            emotion_scores[emotion] += confidence
            total_confidence += confidence
            total_faces += 1

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{emotion} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)

        # Save the annotated image
        try:
            annotated_path = os.path.join(faces_dir, f"annotated_{os.path.basename(file_path)}")
            plt.figure(figsize=(10, 5))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.savefig(annotated_path)
            plt.close()  # Close figure to free memory
        except Exception as e:
            print(f"[WARNING] Failed to save or render annotated image for {file_path}: {e}")
            annotated_path = None

        print(f"Annotated face image saved at: {annotated_path}")

        if total_faces == 0 or total_confidence == 0:
            return None, None, None, None

        weighted_avg_emotion = {emotion: score / total_confidence for emotion, score in emotion_scores.items()}
        dominant_emotion = max(weighted_avg_emotion, key=weighted_avg_emotion.get)
        dominant_sentiment = self.get_sentiment_from_emotion(dominant_emotion)
        return dominant_emotion, weighted_avg_emotion[dominant_emotion], dominant_sentiment, annotated_path


    @staticmethod
    def get_sentiment_from_emotion(emotion:str):
        """
        Mapping of the emotions to sentiment.
        """
        emotion_to_sentiment = {
            "happy": "positive",
            "sad": "negative",
            "surprise": "positive",
            "angry": "negative",
            "neutral": "neutral",
            "disgust": "negative",
            "fear": "negative",
            "anger": "negative",
            "joy": "positive",
            "sadness": "negative"
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