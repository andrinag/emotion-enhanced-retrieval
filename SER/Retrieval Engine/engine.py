import os
from http.client import HTTPException
from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
import psycopg2
import io
from pgvector.psycopg2 import register_vector
from transformers import CLIPProcessor, CLIPModel
import uvicorn
import cv2


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") # pretrained CLIP model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
FRAME_STORAGE = "./frames" # local storage for the video frames
if not os.path.exists(FRAME_STORAGE):
    os.makedirs(FRAME_STORAGE)

app = FastAPI()

# connection to the local database
conn = psycopg2.connect(
    dbname="bachelorthesis",
    user="postgres",
    password="123",
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
            inputs = processor(text=input_text, return_tensors="pt", padding=True)
            return model.get_text_features(**inputs).numpy().flatten()
        elif input_image:
            image = Image.open(io.BytesIO(input_image)).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            print(model.get_image_features(**inputs).numpy().flatten())
            return model.get_image_features(**inputs).numpy().flatten()


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
            image = Image.open(io.BytesIO(file_content)).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            embedding = model.get_image_features(**inputs).detach().numpy().flatten()

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
    frame_count = 0
    success, image = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * seconds)

    while success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        success, image = cap.read()

        if success:
            frame_filename = f"{output_folder}/frame_{frame_count // frame_interval}.jpg"
            cv2.imwrite(frame_filename, image)
            frame_count += frame_interval
        else:
            break

    cap.release()
    print(f"Frames extracted to {output_folder}")


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
            extract_frames(video_path, FRAME_STORAGE)

            for filename in sorted(os.listdir(FRAME_STORAGE)):
                if not filename.endswith((".jpg", ".png")):
                    continue

                frame_time = int(filename.split("_")[-1].split(".")[0])
                image = Image.open(os.path.join(FRAME_STORAGE, filename)).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
                embedding = model.get_image_features(**inputs).detach().numpy().flatten()

                # store every thing (frame and embedding) in DB
                cursor.execute(
                    "INSERT INTO multimedia_embeddings (object_id, frame_time, embedding) VALUES (%s, %s, %s);",
                    (object_id, frame_time, embedding.tolist())
                )

            conn.commit()
            print(f"Stored frames from {file.filename} in DB.")

        return {"message": f"Uploaded and processed {len(files)} videos successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


################## TEXT TO IMAGE SEARCH #########################
@app.get("/search/{query}")
async def search_images(query: str):
    """
    allows you to search for similar images via query (text) input
    :param query: string
    :return: returns the 3 closest images for the query
    """
    try:
        cursor = conn.cursor()
        query_embedding = get_embedding(input_text=query)
        # TODO: doesn't fully work, need to figure how to make it better
        cursor.execute("""
            SELECT mo.location, me.frame_time, me.embedding <-> %s::vector AS distance
            FROM multimedia_embeddings me
            JOIN multimedia_objects mo ON me.object_id = mo.object_id
            ORDER BY distance ASC
            LIMIT 3;
        """, (query_embedding.tolist(),))

        results = cursor.fetchall()
        cursor.close()
        return {
            "results": [
                {
                    "location": row[0],
                    "frame_time": row[1] if row[1] is not None else None,
                    "distance": row[2]
                }
                for row in results
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)