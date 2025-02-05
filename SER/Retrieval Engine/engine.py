import os
from http.client import HTTPException
from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
import psycopg2
import io
from pgvector.psycopg2 import register_vector
import uvicorn
import cv2
import open_clip

# load the clip model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

FRAME_STORAGE = "./frames" # local storage for the video frames
if not os.path.exists(FRAME_STORAGE):
    os.makedirs(FRAME_STORAGE)

app = FastAPI()
BATCH_SIZE = 10

# connection to the local database
conn = psycopg2.connect(
    dbname="multimedia_db",
    user="postgres",
    host="localhost",
    port="5432"
)
register_vector(conn)


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


###################### UPLOADING IMAGES ##############################

def insert_image_metadata(filename):
    """
    inserts the metadata of the image into the multimedia_object table
    :param filename: name of the image file
    :return: object_id of the just inserted image tuple
    """
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO multimedia_objects (type, location) VALUES (%s, %s) RETURNING object_id;",
                       ("image", filename))
        object_id = cursor.fetchone()[0]
        conn.commit()
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
        cursor = conn.cursor()
        for file in files:
            file_content = file.file.read()
            file.file.seek(0)
            embedding = get_embedding(input_image=file_content)

            # first insert the image and its meatdata (file, id etc.)
            object_id = insert_image_metadata(file.filename)
            cursor.execute(
                "INSERT INTO multimedia_embeddings (object_id, frame_time, embedding) VALUES (%s, NULL, %s);",
                (object_id, embedding.tolist())
            )

        conn.commit()
        return {"message": f"Uploaded {len(files)} images successfully"}

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        cursor.close()


######################## UPLOADING VIDEO ################################

def extract_frames(video_path, output_folder, seconds=1):
    """
    extracts frames from a video in given interval (here 1s)
    :param video_path: path to where the video file is stored
    :param output_folder: path to where frames should be stored
    :param seconds: in what interval frames should be taken
    :return:
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    success, image = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * seconds) if fps > 0 else 1
    frame_count = 0
    frame_paths = []

    while success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        success, image = cap.read()
        if success:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, image)
            frame_paths.append((frame_filename, frame_count // frame_interval))
            frame_count += frame_interval
        else:
            break

    cap.release()
    return frame_paths


def insert_video_metadata(video_filename):
    """
    inserts the metadata of the video into the multimedia_object table
    :param video_filename: name of the video file
    :return: object_id of the just inserted video tuple
    """
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO multimedia_objects (type, location) VALUES (%s, %s) RETURNING object_id;",
                       ("video", video_filename))
        object_id = cursor.fetchone()[0]
        conn.commit()
        return object_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_videos/")
async def process_videos(files: list[UploadFile] = File(...)):
    """
    takes list of videos, makes frames of them, caculates embedding and then stores it in the DB
    :param files: video files to be stored
    :return: status code
    """
    cursor = conn.cursor()
    try:
        for file in files:
            # first save the video locally to get frames and embedding
            video_path = os.path.join(FRAME_STORAGE, file.filename)
            with open(video_path, "wb") as f:
                f.write(file.file.read())
            object_id = insert_video_metadata(file.filename)
            frame_data = extract_frames(video_path, FRAME_STORAGE)
            batch = []

            for frame_path, frame_time in frame_data:
                image = Image.open(frame_path).convert("RGB")
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="PNG")
                img_byte_arr = img_byte_arr.getvalue()
                embedding = get_embedding(input_image=img_byte_arr)
                batch.append((object_id, frame_time, embedding.tolist()))
                if (len(batch)) >= BATCH_SIZE:
                    cursor.executemany(
                        "INSERT INTO multimedia_embeddings (object_id, frame_time, embedding) VALUES (%s, %s, %s);",
                        batch,
                    )
                    conn.commit()
                    batch.clear()
            # print(f"Stored frames from {file.filename} in DB.")
            # in case there is stuff left in the batch
            if batch:
                cursor.executemany(
                    "INSERT INTO multimedia_embeddings (object_id, frame_time, embedding) VALUES (%s, %s, %s);",
                    batch,
                )
                conn.commit()

        return {"message": f"Uploaded and processed {len(files)} videos successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)