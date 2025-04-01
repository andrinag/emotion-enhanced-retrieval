import json
import os
import psycopg2
import pandas as pd
import re
from pathlib import Path
from collections import defaultdict
from sentiment_detector import SentimentDetector
import cv2

conn = psycopg2.connect(
    dbname="multimedia_db",
    user="test",
    password="123",
    host="localhost",
    port=5433
)
cursor = conn.cursor()

def insert_ocr(json_path):
    tsv_base_path = Path("/home/ubuntu/V3C1_msb/msb")
    tsv_base_path = Path("./V3C1_msb/msb")
    SD = SentimentDetector()
    tsv_cache = {}
    object_id_cache = {}
    embedding_id_cache = {}

    def get_frame_from_tsv(video_id, row_index):
        if video_id not in tsv_cache:
            tsv_path = tsv_base_path / f"{video_id}.tsv"
            if not tsv_path.exists():
                print(f"TSV not found for video {video_id}")
                return None
            tsv_df = pd.read_csv(tsv_path, sep="\t", names=["startframe", "starttime", "endframe", "endtime"])
            tsv_cache[video_id] = tsv_df
        tsv_df = tsv_cache[video_id]
        if row_index >= len(tsv_df):
            return None
        row = tsv_df.iloc[row_index]
        return (int(row["startframe"]) + int(row["endframe"])) // 2

    image_detections = defaultdict(list)

    with open(json_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                for det in entry.get("detections", []):
                    image_name = det.get("image", "")
                    image_detections[image_name].append(det)
            except json.JSONDecodeError:
                continue

    for image_name, detections in image_detections.items():
        match = re.match(r"v_(\d+)_(\d+)\.jpg", image_name)
        if not match:
            continue

        video_id, row_index_str = match.groups()
        row_index = int(row_index_str)
        frame = get_frame_from_tsv(video_id, row_index)
        if frame is None:
            continue

        if video_id not in object_id_cache:
            cursor.execute("SELECT object_id FROM multimedia_objects WHERE location LIKE %s", (f"%{video_id}.mp4%",))
            result = cursor.fetchone()
            if not result:
                continue
            object_id_cache[video_id] = result[0]
        object_id = object_id_cache[video_id]

        key = (object_id, frame)
        if key not in embedding_id_cache:
            cursor.execute("SELECT id FROM multimedia_embeddings WHERE object_id = %s AND frame = %s", (object_id, frame))
            result = cursor.fetchone()
            if not result:
                continue
            embedding_id_cache[key] = result[0]
        embedding_id = embedding_id_cache[key]

        image_path = get_location_for_frame(embedding_id)
        annotated_path = draw_all_detections_on_image(image_path, detections)

        for det in detections:
            text = det.get("text", "")
            conf = det.get("confidence", 0.0)

            x = det.get("x")
            y = det.get("y")
            w = det.get("w")
            h = det.get("h")

            sentiment_result = SD.get_emotion_from_text(text)
            if sentiment_result and isinstance(sentiment_result, list) and len(sentiment_result) > 0:
                top_emotion = sentiment_result[0]['label']
                sentiment_confidence = sentiment_result[0]['score']
                sentiment_category = SD.get_sentiment_from_emotion(top_emotion)
            else:
                top_emotion = "neutral"
                sentiment_confidence = 0.0
                sentiment_category = "neutral"

            cursor.execute("""
                INSERT INTO OCR (embedding_id, text, emotion, sentiment_confidence, ocr_confidence, sentiment, path_annotated_location, 
                                 x, y, w, h)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                embedding_id,
                text,
                top_emotion,
                sentiment_confidence,
                conf,
                sentiment_category,
                annotated_path,
                x, y, w, h
            ))
            conn.commit()
            #print(f"Inserted embedding for {embedding_id} - done.")


def draw_all_detections_on_image(image_path, detections):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f" Image not found: {image_path}")
            return image_path

        height, width = image.shape[:2]

        for det in detections:
            text = det.get("text", "")
            relX = det.get("relX", 0.0)
            relY = det.get("relY", 0.0)
            relW = det.get("relW", 0.0)
            relH = det.get("relH", 0.0)

            w = relW * width
            h = relH * height
            x = relX * width
            y = (1.0 - relY - relH) * height  # <-- flip Y because Vision origin is bottom-left

            x1 = int(x)
            y1 = int(y)
            x2 = int(x + w)
            y2 = int(y + h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, text, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        output_dir = "./ocr_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, image)
        # print(out_path)
        return out_path
    except Exception as e:
        print(f"Failed drawing on {image_path}: {e}")
        return image_path


def get_ocr_list_from_folder(folder_path):
    return [
        os.path.abspath(os.path.join(folder_path, file))
        for file in os.listdir(folder_path)
        if file.endswith(".jsonl")
    ]

def get_location_for_frame(embedding_id):
    try:
        cursor.execute("SELECT frame_location FROM multimedia_embeddings WHERE id = %s;", (embedding_id,))
        result = cursor.fetchone()
        return result[0] if result else None
    except Exception as e:
        print(f"Could not get frame location for {embedding_id}: {e}")


if __name__ == "__main__":
    folder_path = "./OCR_V3C1"
    ocr_files_list = get_ocr_list_from_folder(folder_path)
    for ocr_json in ocr_files_list[:1]:
        print(f"Currently working on: {ocr_json}")
        insert_ocr(ocr_json)
