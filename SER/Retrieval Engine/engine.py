from fastapi import FastAPI
import torch
from PIL import Image
import psycopg2
import io
import uvicorn
import cv2
import open_clip
from concurrent.futures import ThreadPoolExecutor
from fastapi import UploadFile, File, HTTPException
import os
import traceback
import numpy as np
import pandas as pd
import ffmpeg
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from sentiment_detector import SentimentDetector
from psycopg2 import pool

MAX_WORKERS = 16
BATCH_SIZE = 50
FRAME_STORAGE = "./frames"

# load the clip model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

#  local storage of the video frames
FRAME_STORAGE = "./frames"
if not os.path.exists(FRAME_STORAGE):
    os.makedirs(FRAME_STORAGE)

mastershot_dir_1 = "/home/ubuntu/V3C1_msb/msb"
# mastershot_dir_1  = "./V3C1_msb/msb"

app = FastAPI()

# SD = SentimentDetector()

db_pool = psycopg2.pool.SimpleConnectionPool(
    1, 20,  # Min and max connections
    dbname="multimedia_db",
    user="test",
    password="123",
    host="localhost",
    port="5432"
)


def get_db_connection():
    """
    returns multiple DB connections
    """
    return db_pool.getconn()


def release_db_connection(conn):
    """
    closes multiple db connections
    """
    db_pool.putconn(conn)


def get_embedding(input_text=None, input_image=None):
    """
    calculates the embedding of text and images with CLIP
    :param input_text: input text to embed
    :param input_image: input image to embed
    :return: returns the embedded values
    """
    with torch.no_grad():
        if input_text:
            inputs = tokenizer([input_text])
            features = model.encode_text(inputs)
            return features.numpy().flatten()
        elif input_image:
            image = Image.open(io.BytesIO(input_image)).convert("RGB")
            inputs = preprocess(image).unsqueeze(0)
            features = model.encode_image(inputs)
            return features.numpy().flatten()


def normalize_embedding(embedding):
    """
    normalizes the embedding
    """
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def check_file_exists(filename):
    """
    checks if a file with the same name already exists in the database
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT object_id FROM multimedia_objects WHERE location = %s;", (filename,))
        result = cursor.fetchone()
        return result[0] if result else None
    finally:
        cursor.close()
        release_db_connection(conn)


###################### UPLOADING IMAGES ##############################

def insert_image_metadata(filename):
    """
    inserts the metadata of the image into the multimedia_object table
    :param filename: name of the image file
    :return: object_id of the just inserted image tuple
    """
    try:
        cursor = get_db_connection().cursor()
        cursor.execute("INSERT INTO multimedia_objects (location) VALUES (%s) RETURNING object_id;",
                       (filename,))
        object_id = cursor.fetchone()[0]
        get_db_connection().commit()
        return object_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_image/")
async def upload_files(files: list[UploadFile] = File(...)):
    """
    uploads the local images form a list into the DB with their embeddings
    :param files: list of image files to upload
    :return: status of request
    """
    try:
        cursor = get_db_connection().cursor()
        for file in files:
            file_content = file.file.read()
            file.file.seek(0)
            embedding = get_embedding(input_image=file_content)
            embedding = normalize_embedding(embedding)

            # first insert the image and its meatdata (file, id etc.)
            object_id = insert_image_metadata(file.filename)
            cursor.execute(
                "INSERT INTO multimedia_embeddings (object_id, frame_time, embedding) VALUES (%s, NULL, %s);",
                (object_id, embedding.tolist())
            )

        get_db_connection().commit()
        return {"message": f"Uploaded {len(files)} images successfully"}

    except Exception as e:
        get_db_connection().rollback()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        cursor.close()


######################## UPLOADING VIDEO ################################

def extract_frames(video_path, output_folder, msb_file):
    """
    extract frame per mastershot boundary. takes the middle part.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        boundaries = pd.read_csv(msb_file, sep="\t")  # in tsv files \t is the seperator
    except Exception as e:
        print(f"Error reading boundary file {msb_file}: {e}")
        return []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print(f"Error: FPS is zero for:  {video_path}")
        return []

    frame_paths = []

    for index, row in boundaries.iterrows():
        start_frame = int(row['startframe'])
        end_frame = int(row['endframe'])
        start_time = row['starttime']
        end_time = row['endtime']

        # because we take the middle point of the segment of the msb
        middle_frame = (start_frame + end_frame) // 2
        middle_time = (start_time + end_time) / 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame) # sets it to the middle point
        success, image = cap.read()
        if not success:
            print(f"Warning: Could not extract frame {middle_frame} from {video_path}")
            continue
        frame_filename = os.path.join(output_folder, f"{os.path.basename(video_path)}_frame_{middle_frame}.jpg")
        cv2.imwrite(frame_filename, image)
        frame_paths.append((frame_filename, middle_time, middle_frame))

    cap.release()
    return frame_paths


def insert_video_metadata(video_filename):
    """
    inserts the metadata of the video into the multimedia_object table
    :param video_filename: name of the video file
    :return: object_id of the just inserted video tuple
    """
    cursor = get_db_connection().cursor()
    try:
        cursor.execute("INSERT INTO multimedia_objects (location) VALUES (%s) RETURNING object_id;",
                       (video_filename,))
        object_id = cursor.fetchone()[0]
        get_db_connection().commit()
        return object_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def process_frame(frame_info, object_id):
    """
    processes a single video frame and calls the embedding calculation method
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        frame_path, frame_time, middle_frame = frame_info
        image = Image.open(frame_path).convert("RGB")
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        embedding = get_embedding(input_image=img_byte_arr.getvalue()) # calls the methods that uses CLIP
        embedding = normalize_embedding(embedding)

        cursor.execute(
            """
            INSERT INTO multimedia_embeddings (object_id, frame, frame_time, embedding)
            VALUES (%s, %s, %s, %s) RETURNING id;
            """,
            (object_id, int(middle_frame), float(frame_time), embedding.tolist())
        )
        embedding_id = cursor.fetchone()[0]
        conn.commit()
        SD = SentimentDetector()
        emotion, confidence, sentiment, annotated_path = SD.detect_faces_and_get_emotion_with_plots(frame_path)

        cursor.execute(
            """
            INSERT INTO Face (embedding_id, emotion, confidence, sentiment, path_annotated_faces)
            VALUES (%s, %s, %s, %s, %s);
            """,
            (embedding_id, emotion, confidence, emotion, annotated_path)
        )
        conn.commit()
    except Exception as e:
        print("Error processing frame:", traceback.format_exc())
        conn.rollback()
    finally:
        cursor.close()
        release_db_connection(conn)


@app.post("/upload_videos/")
async def process_videos(files: list[UploadFile] = File(...)):
    """
    takes a list of videos and calls the corresponding methods to extract the frames, calculate the embedding and
    insert into the DB
    """
    try:
        for file in files:
            existing_object_id = check_file_exists(file.filename)
            if existing_object_id:
                print(f"Skipping {file.filename}: Already exists in the database.")
                continue
            SD = SentimentDetector()

            video_filename = file.filename
            video_path = os.path.join(FRAME_STORAGE, video_filename)
            with open(video_path, "wb") as f:
                f.write(file.file.read())
            conn = get_db_connection()
            cursor = conn.cursor()
            # Todo change this to the metadata method
            cursor.execute("INSERT INTO multimedia_objects (location) VALUES (%s) RETURNING object_id;",
                           (file.filename,))
            object_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            release_db_connection(conn)

            boundary_file = os.path.splitext(mastershot_dir_1)[0] + "/" + video_filename.split(".")[0] + ".tsv"
            if not os.path.exists(boundary_file):
                print(f"Skipping {video_filename}: No boundary file found ({boundary_file})")
                continue

            frame_data = extract_frames(video_path, FRAME_STORAGE, boundary_file)
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                executor.map(lambda f: process_frame(f, object_id), frame_data)

        return {"message": f"Uploaded and processed {len(files)} videos successfully"}

    except Exception as e:
        error_message = traceback.format_exc()
        print("Error Traceback:", error_message)
        return {"error": str(e), "traceback": error_message}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)