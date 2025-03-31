import json
import os
import psycopg2
import pandas as pd
import re
from pathlib import Path
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
    tsv_base_path = Path("V3C1_msb/msb")

    tsv_cache = {}
    object_id_cache = {}
    embedding_id_cache = {}

    SD = SentimentDetector()

    def get_frame_from_tsv(video_id, row_index):
        """Get frame from specified row index of a video’s TSV file."""
        row_index = row_index
        if video_id not in tsv_cache:
            tsv_path = tsv_base_path / f"{video_id}.tsv"
            if not tsv_path.exists():
                print(f"TSV not found for video {video_id}")
                return None
            tsv_df = pd.read_csv(tsv_path, sep="\t", names=["startframe", "starttime", "endframe", "endtime"])
            tsv_cache[video_id] = tsv_df

        tsv_df = tsv_cache[video_id]
        if row_index >= len(tsv_df):
            print(f"TSV for video {video_id} has no row {row_index}")
            return None

        row = tsv_df.iloc[row_index]
        return (int(row["startframe"]) + int(row["endframe"])) // 2

    with open(json_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print("Skipping invalid JSON line")
                continue

            for detection in entry.get("detections", []):
                image_name = detection.get("image", "")
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
                        print(f"No multimedia_object found for {video_id}.mp4")
                        continue
                    object_id_cache[video_id] = result[0]

                object_id = object_id_cache[video_id]

                key = (object_id, frame)
                if key not in embedding_id_cache:
                    cursor.execute("SELECT id FROM multimedia_embeddings WHERE object_id = %s AND frame = %s", (object_id, frame))
                    result = cursor.fetchone()
                    if not result:
                        print(f"No embedding found for object_id {object_id} and frame {frame}")
                        continue
                    embedding_id_cache[key] = result[0]

                embedding_id = embedding_id_cache[key]

                ocr_confidence = detection.get("confidence", 0.0)
                text = detection.get("text", "")
                path_annotated_location = detection.get("image", "")
                x1 = detection.get("relX", 0.0)
                y1 = detection.get("relY", 0.0)
                w = detection.get("relW", 0.0)
                h = detection.get("relH", 0.0)
                x2 = x1 + w
                y2 = y1 + h

                frame_location = get_location_for_frame(embedding_id)
                saved_ocr_annotations = draw_ocr_visualization(x1, x2, y1, y2, frame_location)
                sentiment_result = SD.get_emotion_from_text(text)
                if sentiment_result and isinstance(sentiment_result, list) and len(sentiment_result) > 0:
                    top_emotion = sentiment_result[0]['label']
                    sentiment_confidence = sentiment_result[0]['score']
                    sentiment_category = SD.get_sentiment_from_emotion(top_emotion)
                    emotion = "neutral"
                    sentiment = "neutral"

                cursor.execute("""
                    INSERT INTO OCR (embedding_id, text, emotion, sentiment_confidence, ocr_confidence, sentiment, path_annotated_location, 
                                     y_axis1, x_axis1, y_axis2, x_axis2)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (embedding_id, str(text), top_emotion, sentiment_confidence, ocr_confidence, sentiment, saved_ocr_annotations,
                      y1, x1, y2, x2))

    conn.commit()
    cursor.close()
    conn.close()
    print("OCR data inserted successfully using row index extracted from image filename.")


def draw_ocr_visualization(rel_x, rel_y, rel_w, rel_h, frame_location, label="OCR"):
    try:
        image = cv2.imread(frame_location)
        if image is None:
            print(f"[WARNING] Could not read image: {frame_location}")
            return None

        height, width = image.shape[:2]

        # Convert to absolute pixel values
        x1 = int(rel_x * width)
        y1 = int(rel_y * height)
        x2 = int((rel_x + rel_w) * width)
        y2 = int((rel_y + rel_h) * height)

        # Correct mirrored or flipped coordinates
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        # Clip to image bounds
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        # Draw rectangle and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Save visualization
        output_dir = "./ocr_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(frame_location)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, image)

        return output_path

    except Exception as e:
        print(f"[ERROR] Could not draw box on {frame_location}: {e}")
        return None



def get_location_for_frame(embedding_id):
    try:
        cursor.execute("SELECT frame_location FROM multimedia_embeddings WHERE id = %s;", (embedding_id,))
        result = cursor.fetchone()
        print(result)
        return result[0] if result else None
    except Exception as e:
        print(f"[ERROR] Could not get frame location for {embedding_id}: {e}")

def get_ocr_list_from_folder(folder_path):
    video_list = [
        os.path.abspath(os.path.join(folder_path, file))
        for file in os.listdir(folder_path)
        if file.endswith(".jsonl")
    ]
    return video_list

if __name__ == "__main__":
    folder_path = "./OCR_V3C1" # local
    ocr_files_list = get_ocr_list_from_folder(folder_path)

    for ocr_json in ocr_files_list[:1]:
        insert_ocr(ocr_json)