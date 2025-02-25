import os
from http.client import HTTPException
import cv2
from fastapi import FastAPI, UploadFile, File, Response, Header, Request, HTTPException
import torch
from PIL import Image
import psycopg2
import io
from pgvector.psycopg2 import register_vector
import uvicorn
from fastapi.staticfiles import StaticFiles
import open_clip
from pathlib import Path
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import numpy as np

# load the clip model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

templates = Jinja2Templates(directory="templates")
CHUNK_SIZE = 1024*1024

# local storage for the video frames (images)
FRAME_STORAGE = "./frames"
if not os.path.exists(FRAME_STORAGE):
    os.makedirs(FRAME_STORAGE)

app = FastAPI()
app.mount("/media/V3C/V3C1/video-480p/", StaticFiles(directory="/media/V3C/V3C1/video-480p/"), name="videos")
#locally
# app.mount("/videos", StaticFiles(directory="./videos"), name="videos")


# connection to the local database
conn = psycopg2.connect(
    dbname="multimedia_db",
    user="test",
    host="localhost",
    password="123",
    port="5432"
)
register_vector(conn)

def normalize_embedding(embedding):
    """
    Brings the embeddings into a normalized form
    """
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


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


################## IMAGE TO IMAGE SEARCH #################################################
@app.post("/search_image_to_image/")
async def search_image_to_image(file: UploadFile = File(...)):
    """
    Allows you to search the closest image in the database compared to a given image
    """
    cursor = conn.cursor()
    try:
        file_content = file.file.read()
        file.file.seek(0)
        embedding = get_embedding(input_image=file_content)
        embedding = normalize_embedding(embedding)
        # Important. Cosine comparison
        cursor.execute("""
            SELECT 
                (SELECT location FROM multimedia_objects WHERE object_id = me.object_id) AS location,
                me.frame_time,
                1 - (me.embedding <=> %s::vector) AS similarity
            FROM multimedia_embeddings me
            ORDER BY similarity DESC
            LIMIT 5; 
            """, (embedding.tolist(),))

        result = cursor.fetchall()
        for row in result:
            print(f"Video: {row[0]}, Frame Time: {row[1]}, Similarity: {row[2]}")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        cursor.close()
################## TEXT TO IMAGE SEARCH #########################
@app.get("/search/{query}")
async def search_images(request: Request, query: str):
    """
    Search for videos related to the query and return the video path.
    TODO: make normalization of the result values for text to image
    """
    try:
        cursor = conn.cursor()
        query_embedding = get_embedding(input_text=query)
        query_embedding = normalize_embedding(query_embedding)
        print(query_embedding[-10:])

    # Cosine comparison for the similarity
        cursor.execute("""
        SELECT 
            (SELECT location FROM multimedia_objects WHERE object_id = me.object_id) AS location,
            me.frame_time,
            1 - (me.embedding <=> %s::vector) AS similarity
        FROM multimedia_embeddings me
        ORDER BY similarity DESC
        LIMIT 5; 
        """, (query_embedding.tolist(),))

        result = cursor.fetchall()
        for row in result:
            print(f"Video: {row[0]}, Frame Time: {row[1]}, Similarity: {row[2]}")

        cursor.close()

        if result:
            response = [
                {
                    "video_path": f"/media/V3C/V3C1/video-480p/{row[0]}",
                    "frame_time": row[1],
                    "similarity": row[2]
                }
                for row in result
            ]
            print(response)
            return JSONResponse(content=response, media_type="application/json")
        else:
            return JSONResponse({"message": "No video found"}, status_code=404)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})

@app.get("/video")
async def video_endpoint(path: str, start_time: float = 0.0, range: str = Header(None)):
    """
    Allows a webbrowser to play the video which is stored locally
    the fallback directory is the path to the V3C2 directory, because I made a mistake in my DB schema
    """
    file_path = Path(path)
    #fallback_directory = "./videos"
    fallback_directory = "/media/V3C/V3C2/video-480p/"
    fallback_path = Path(fallback_directory) / file_path.name

    def get_byte_offset(video_path, start_time):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if fps <= 0:
            raise HTTPException(status_code=500, detail="Failed to retrieve video FPS")

        start_frame = int(start_time * fps)
        file_size = video_path.stat().st_size
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        avg_bytes_per_frame = file_size / total_frames if total_frames > 0 else CHUNK_SIZE

        return int(start_frame * avg_bytes_per_frame)

    def stream_video(video_path, start_time):
        byte_offset = get_byte_offset(video_path, start_time)

        if range:
            start, end = range.replace("bytes=", "").split("-")
            start = int(start)
            end = int(end) if end else start + CHUNK_SIZE
        else:
            start = byte_offset
            end = start + CHUNK_SIZE

        with open(video_path, "rb") as video:
            video.seek(start)
            data = video.read(end - start)
            filesize = str(video_path.stat().st_size)

            headers = {
                'Content-Range': f'bytes {start}-{start + len(data) - 1}/{filesize}',
                'Accept-Ranges': 'bytes',
                'Content-Length': str(len(data)),
                'Content-Type': 'video/mp4'
            }

            return Response(data, status_code=206, headers=headers, media_type="video/mp4")
    if file_path.exists():
        return stream_video(file_path, start_time)
    if fallback_path.exists():
        return stream_video(fallback_path, start_time)

    raise HTTPException(status_code=404, detail="Video not found")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
