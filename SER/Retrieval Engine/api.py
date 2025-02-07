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
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import open_clip
from starlette.responses import StreamingResponse
from pathlib import Path
from fastapi import FastAPI
from fastapi import Request, Response
from fastapi import Header
from fastapi.templating import Jinja2Templates
from fastapi import HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse

# load the clip model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')
templates = Jinja2Templates(directory="templates")
CHUNK_SIZE = 1024*1024
video_path = Path("./videos/seafood_1280p.mp4")

FRAME_STORAGE = "./frames" # local storage for the video frames
if not os.path.exists(FRAME_STORAGE):
    os.makedirs(FRAME_STORAGE)

app = FastAPI()
app.mount("/videos", StaticFiles(directory="videos"), name="videos")


# connection to the local database
conn = psycopg2.connect(
    dbname="multimedia_db",
    user="test",
    host="localhost",
    password="123",
    port="5433"
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


################## TEXT TO IMAGE SEARCH #########################
@app.get("/search/{query}")
async def search_images(request: Request, query: str):
    """
    Search for videos related to the query and return the video path.
    """
    try:
        cursor = conn.cursor()
        query_embedding = get_embedding(input_text=query)

        cursor.execute("""
            SELECT mo.location, me.frame_time, me.embedding <-> %s::vector AS distance
            FROM multimedia_embeddings me
            JOIN multimedia_objects mo ON me.object_id = mo.object_id
            ORDER BY distance ASC
            LIMIT 1;
        """, (query_embedding.tolist(),))

        result = cursor.fetchone()
        cursor.close()

        if result:
            video_path = result[0]
            return JSONResponse({"video_path": f"./videos/{video_path}", "frame_time": result[1], "distance": result[2]})
        else:
            return JSONResponse({"message": "No video found"}, status_code=404)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})


@app.get("/video")
async def video_endpoint(path: str, range: str = Header(None)):
    video_path = Path(path)

    try:
        start, end = range.replace("bytes=", "").split("-")
        start = int(start)
        end = int(end) if end else start + CHUNK_SIZE

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
    except Exception as e:
        raise HTTPException(status_code=404, detail="Video not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
