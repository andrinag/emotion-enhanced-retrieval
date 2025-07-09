import os
import traceback
from http.client import HTTPException
import cv2
import requests
from fastapi import FastAPI, UploadFile, File, Response, Header, Request, HTTPException
import torch
from PIL import Image
import psycopg2
import io
from pgvector.psycopg2 import register_vector
import uvicorn
import open_clip
from pathlib import Path
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from sklearn.utils import deprecated

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
app.mount("/faces", StaticFiles(directory="./faces"), name="faces")
app.mount("/frames", StaticFiles(directory="./frames"), name="frames")
app.mount("/ocr_visualizations", StaticFiles(directory="./ocr_visualizations"), name="ocr_visualizations")
#locally
# app.mount("/videos", StaticFiles(directory="./videos"), name="videos")


# connection to the local database
conn = psycopg2.connect(
    dbname="multimedia_db",
    user="test",
    host="10.34.64.139",
    password="123",
    port="5432"
)
register_vector(conn)

dir_1 = "/media/V3C/V3C1/video-480p/"

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


VIDEO_DIRECTORY = "/media/V3C/V3C1/video-480p/"
fallback_dir = "/media/V3C/V3C2/video-480p/"

@app.api_route("/media/V3C/{subfolder}/video-480p/{video_name}", methods=["GET", "HEAD"])
async def video_endpoint(subfolder: str, video_name: str, range: str = Header(None)):
    """
    serves videos for http requests ( relevant for the app, not the website)
    """
    file_path = Path(f"/media/V3C/{subfolder}/video-480p/") / video_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    file_size = file_path.stat().st_size

    with open(file_path, "rb") as video:
        if range:
            start, end = range.replace("bytes=", "").split("-")
            start = int(start) if start else 0
            end = int(end) if end else min(start + 1024 * 1024, file_size - 1)

            video.seek(start)
            data = video.read(end - start + 1)

            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(len(data)),
                "Content-Type": "video/mp4",
            }

            return Response(data, status_code=206, headers=headers, media_type="video/mp4")

        data = video.read()
        headers = {
            "Content-Length": str(file_size),
            "Content-Type": "video/mp4",
            "Accept-Ranges": "bytes",
        }
        return Response(data, status_code=200, headers=headers, media_type="video/mp4")

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


#######################################################
#                 TEXT TO IMAGE SEARCH                #
#######################################################
@app.get("/search/{query}/{allow_duplicates")
@deprecated # function no longer in use in current implementation. Still useful.
async def search_images(query: str, allow_duplicates: str):
    """
    Search for videos related to the query and return the video path.
    TODO: make normalization of the result values for text to image
    """
    dir_1 = "/media/V3C/V3C1/video-480p/"
    try:
        cursor = conn.cursor()
        query_embedding = get_embedding(input_text=query)
        query_embedding = normalize_embedding(query_embedding)
        #print(query_embedding[-10:])

        if allow_duplicates:
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
        else:
            cursor.execute("""
                SELECT
                    (SELECT DISTINCT location FROM multimedia_objects WHERE object_id = me.object_id) AS location,
                    me.frame_time,
                    1 - (me.embedding <=> %s::vector) AS similarity
                FROM multimedia_embeddings me
                ORDER BY similarity DESC
                LIMIT 5; 
                """, (query_embedding.tolist(),))

            result = cursor.fetchall()
        cursor.close()

        if not result:
            return JSONResponse({"message":"No video found"}, status_code=404)

        response = []
        for row in result:
            print(f"Video: {row[0]}, Frame Time: {row[1]}, Similarity: {row[2]}")
            if os.path.exists(dir_1 + row[0]):
                final_path = dir_1 + row[0]
                response = [
                    {
                        "video_path": final_path,
                        "frame_time": row[1],
                        "similarity": row[2]
                    }
                    for row in result
                ]
            print(response)
            return JSONResponse(response)
        else:
            return JSONResponse({"message": "No video found"}, status_code=404)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


##############################################################################
#                 TEXT TO IMAGE SEARCH WITH EMOTION-ENHANCEMENT               #
##############################################################################
@app.get("/search_combined_face/{query}/{emotion}/{allow_duplicates}")
async def search_combined_face(query: str, emotion: str, allow_duplicates: bool):
    """
    Emotion enhanced search for the face modality. Collects the top 1000 most matching embeddings.
    Then filter for emotions. Emotions are only considered if matching, else skip.
    """
    cursor = conn.cursor()

    try:
        emotion_filter = emotion.lower()
        query_filter = query.lower()
        query_embedding = get_embedding(input_text=query_filter)
        query_embedding = normalize_embedding(query_embedding)

        cursor.execute("""
            WITH top_embeddings AS (
                SELECT 
                    me.id AS embedding_id,
                    mo.location,
                    me.frame_time,
                    me.frame_location,
                    1 - (me.embedding <=> %s::vector) AS similarity
                FROM multimedia_embeddings me
                JOIN multimedia_objects mo ON mo.object_id = me.object_id
                ORDER BY similarity DESC
                LIMIT 1000
            ),
            scored_faces AS (
                SELECT 
                    te.embedding_id,
                    te.location,
                    te.frame_time,
                    te.frame_location,
                    te.similarity,
                    f.emotion,
                    f.confidence,
                    f.path_annotated_faces,
                    CASE
                        WHEN LOWER(f.emotion) = LOWER(%s) THEN 1.0
                        ELSE 0.0
                    END AS emotion_match,
                    ((te.similarity * 0.5) + (f.confidence * 0.5)) AS combined_score
                FROM top_embeddings te
                JOIN Face f ON f.embedding_id = te.embedding_id
            )
            SELECT 
                embedding_id,
                location,
                frame_time,
                frame_location,
                similarity,
                emotion_match,
                combined_score,
                path_annotated_faces,
                emotion,
                confidence
            FROM scored_faces
            WHERE emotion_match = 1.0
            ORDER BY combined_score DESC
            LIMIT 20;
        """, (query_embedding.tolist(), emotion_filter))

        result = cursor.fetchall()
        cursor.close()

        if not result:
            return JSONResponse({"message": "No video found"}, status_code=404)

        response = []
        seen_videos = set()
        for row in result:
            (
                embedding_id,
                location,
                frame_time,
                frame_location,
                similarity,
                sentiment_match,
                final_score,
                annotated_path,
                emotion,
                confidence
            ) = row

            full_path = os.path.join(dir_1, location)

            if not os.path.exists(full_path):
                continue

            if not allow_duplicates:
                if full_path in seen_videos:
                    continue
                seen_videos.add(full_path)

            response.append({
                "embedding_id": embedding_id,
                "video_path": full_path,
                "frame_time": float(frame_time),
                "similarity": round(float(similarity), 3),
                "sentiment_match": float(sentiment_match),
                "final_score": round(float(final_score), 3),
                "annotated_image": annotated_path if annotated_path else None,
                "frame_location" : frame_location,
                "face_emotion": emotion,
                "face_confidence": round(float(confidence), 3) if confidence is not None else None
            })

        return JSONResponse(response)

    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/search_combined_asr/{query}/{emotion}/{allow_duplicates}")
async def search_combined_asr(query: str, emotion: str, allow_duplicates: bool):
    """
        Emotion enhanced search for the ASR modality. Collects the top 1000 most matching embeddings.
        Then filter for emotions. Emotions are only considered if matching, else skip.
        """
    dir_1 = "/media/V3C/V3C1/video-480p/"
    cursor = conn.cursor()
    emotion = emotion_mapping(emotion)
    print(emotion)

    try:
        emotion_filter = emotion.lower()
        query_filter = query.lower()

        # Generate and normalize embedding from the query
        query_embedding = get_embedding(input_text=query_filter)
        query_embedding = normalize_embedding(query_embedding)

        cursor.execute("""
            WITH top_embeddings AS (
                SELECT 
                    me.id AS embedding_id,
                    mo.location,
                    me.frame_time,
                    me.frame_location,
                    1 - (me.embedding <=> %s::vector) AS similarity
                FROM multimedia_embeddings me
                JOIN multimedia_objects mo ON mo.object_id = me.object_id
                ORDER BY similarity DESC
                LIMIT 1000
            ),
            scored_asr AS (
                SELECT 
                    te.embedding_id,
                    te.location,
                    te.frame_time,
                    te.frame_location,
                    te.similarity,
                    COALESCE(a.emotion, '') AS emotion,
                    COALESCE(a.confidence, 0.0) AS confidence,
                    COALESCE(a.sentiment, '') AS sentiment,
                    CASE
                        WHEN LOWER(a.emotion) = LOWER(%s) THEN 1.0
                        ELSE 0.0
                    END AS emotion_match,
                    ((te.similarity * 0.5) + (COALESCE(a.confidence, 0.0) * 0.5)) AS combined_score
                FROM top_embeddings te
                LEFT JOIN ASR a ON a.embedding_id = te.embedding_id
            )
            SELECT 
                embedding_id,
                location,
                frame_time,
                frame_location,
                similarity,
                emotion_match,
                combined_score,
                emotion,
                confidence,
                sentiment
            FROM scored_asr
            WHERE emotion_match = 1.0
            ORDER BY combined_score DESC
            LIMIT 20;
        """, (query_embedding.tolist(), emotion_filter))

        result = cursor.fetchall()
        print(result)
        cursor.close()

        if not result:
            return JSONResponse({"message": "No video found"}, status_code=404)

        response = []
        seen_videos = set()
        for row in result:
            (
                embedding_id,
                location,
                frame_time,
                frame_location,
                similarity,
                sentiment_match,
                final_score,
                emotion,
                confidence,
                sentiment_label
            ) = row

            full_path = os.path.join(dir_1, location)

            if not os.path.exists(full_path):
                continue

            if not allow_duplicates:
                if full_path in seen_videos:
                    continue
                seen_videos.add(full_path)

            response.append({
                "embedding_id": embedding_id,
                "video_path": full_path,
                "frame_time": float(frame_time),
                "frame_location": frame_location,
                "similarity": round(float(similarity), 3),
                "sentiment_match": float(sentiment_match),
                "final_score": round(float(final_score), 3),
                "asr_emotion": emotion,
                "asr_confidence": round(float(confidence), 3) if confidence is not None else None,
                "asr_sentiment": sentiment_label
            })
        print(response)
        return JSONResponse(response)

    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/search_combined_ocr/{query}/{emotion}/{allow_duplicates}")
async def search_combined_ocr(query: str, emotion: str, allow_duplicates: bool):
    """
    Emotion enhanced search for the OCR modality. Collects the top 1000 most matching embeddings.
    Then filter for emotions. Emotions are only considered if matching, else skip.
    """
    cursor = conn.cursor()
    emotion = emotion_mapping(emotion)
    print(emotion)

    try:
        query_embedding = get_embedding(input_text=query.lower())
        query_embedding = normalize_embedding(query_embedding)

        cursor.execute("""
            WITH top_embeddings AS (
                SELECT 
                    me.id AS embedding_id,
                    mo.location,
                    me.frame_time,
                    me.frame_location,
                    1 - (me.embedding <=> %s::vector) AS similarity
                FROM multimedia_embeddings me
                JOIN multimedia_objects mo ON mo.object_id = me.object_id
                ORDER BY similarity DESC
                LIMIT 1000
            ),
            scored_ocr AS (
                SELECT 
                    te.embedding_id,
                    te.location,
                    te.frame_time,
                    te.frame_location,
                    te.similarity,
                    COALESCE(o.emotion, '') AS emotion,
                    COALESCE(o.sentiment, '') AS sentiment,
                    COALESCE(o.sentiment_confidence, 0.0) AS sentiment_confidence,
                    COALESCE(o.ocr_confidence, 0.0) AS ocr_confidence,
                    COALESCE(o.path_annotated_location, '') AS path_annotated_location,
                    COALESCE(o.text, '') AS text,
                    COALESCE(o.x, 0.0) AS x,
                    COALESCE(o.y, 0.0) AS y,
                    COALESCE(o.w, 0.0) AS w,
                    COALESCE(o.h, 0.0) AS h,
                    CASE
                        WHEN LOWER(o.emotion) = LOWER(%s) THEN 1.0
                        ELSE 0.0
                    END AS emotion_match,
                    ((te.similarity * 0.5) + (COALESCE(o.sentiment_confidence, 0.0) * 0.5)) AS combined_score
                FROM top_embeddings te
                LEFT JOIN OCR o ON o.embedding_id = te.embedding_id
            )
            SELECT 
                embedding_id,
                location,
                frame_time,
                frame_location,
                similarity,
                emotion_match,
                combined_score,
                path_annotated_location,
                emotion,
                sentiment_confidence,
                ocr_confidence,
                sentiment,
                text,
                x, y, w, h
            FROM scored_ocr
            WHERE emotion_match = 1.0
            ORDER BY combined_score DESC
            LIMIT 20;
        """, (query_embedding.tolist(), emotion.lower()))

        result = cursor.fetchall()
        cursor.close()

        if not result:
            return JSONResponse({"message": "No OCR results found"}, status_code=404)

        response = []
        seen_videos = set()
        for row in result:
            (
                embedding_id,
                location,
                frame_time,
                frame_location,
                similarity,
                sentiment_match,
                final_score,
                annotated_path,
                emotion,
                sentiment_conf,
                ocr_conf,
                sentiment_label,
                ocr_text,
                x, y, w, h
            ) = row

            full_path = os.path.join(dir_1, location)

            if not os.path.exists(full_path):
                continue

            if not allow_duplicates:
                if full_path in seen_videos:
                    continue
                seen_videos.add(full_path)

            response.append({
                "embedding_id": embedding_id,
                "video_path": full_path,
                "frame_time": float(frame_time),
                "similarity": round(float(similarity), 3),
                "sentiment_match": float(sentiment_match),
                "final_score": round(float(final_score), 3),
                "annotated_image": annotated_path if annotated_path else None,
                "frame_location": frame_location,
                "ocr_text": ocr_text,
                "ocr_emotion": emotion,
                "ocr_sentiment": sentiment_label,
                "sentiment_confidence": round(float(sentiment_conf), 3),
                "ocr_confidence": round(float(ocr_conf), 3),
                "bbox": {"x": x, "y": y, "w": w, "h": h}
            })

        return JSONResponse(response)

    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)



@app.get("/search_combined_all/{query}/{emotion}/{allow_duplicates}")
async def search_combined_all(query: str, emotion: str, allow_duplicates: bool):
    """
    Emotion enhanced search for the all 3 modalities modality. Collects the top 1000 most matching embeddings.
    Then filter for emotions and wieghts it with given formula. In most cases not all three modalities can
    have 1 as score.
    """
    cursor = conn.cursor()
    emotion2 = emotion_mapping(emotion)
    print(emotion2)
    try:
        query_embedding = get_embedding(input_text=query.lower())
        query_embedding = normalize_embedding(query_embedding)

        cursor.execute("""
            WITH top_embeddings AS (
                SELECT 
                    me.id AS embedding_id,
                    mo.location,
                    me.frame_time,
                    me.frame_location,
                    1 - (me.embedding <=> %s::vector) AS similarity
                FROM multimedia_embeddings me
                JOIN multimedia_objects mo ON mo.object_id = me.object_id
                ORDER BY similarity DESC
                LIMIT 800
            ),
            joined AS (
                SELECT 
                    te.embedding_id,
                    te.location,
                    te.frame_time,
                    te.frame_location,
                    te.similarity,
                    COALESCE(f.emotion, '') AS face_emotion,
                    COALESCE(f.confidence, 0.0) AS face_confidence,
                    COALESCE(a.emotion, '') AS asr_emotion,
                    COALESCE(a.confidence, 0.0) AS asr_confidence,
                    COALESCE(o.emotion, '') AS ocr_emotion,
                    COALESCE(o.sentiment_confidence, 0.0) AS ocr_confidence,
                    COALESCE(f.path_annotated_faces, '') AS annotated_image,
                    CASE WHEN LOWER(f.emotion) = LOWER(%s) THEN 1 ELSE 0 END AS face_match,
                    CASE WHEN LOWER(a.emotion) = LOWER(%s) THEN 1 ELSE 0 END AS asr_match,
                    CASE WHEN LOWER(o.emotion) = LOWER(%s) THEN 1 ELSE 0 END AS ocr_match
                FROM top_embeddings te
                LEFT JOIN Face f ON f.embedding_id = te.embedding_id
                LEFT JOIN ASR a ON a.embedding_id = te.embedding_id
                LEFT JOIN OCR o ON o.embedding_id = te.embedding_id
            )
            SELECT 
                embedding_id,
                location,
                frame_time,
                frame_location,
                similarity,
                annotated_image,
                face_emotion, face_confidence,
                asr_emotion, asr_confidence,
                ocr_emotion, ocr_confidence,
                (
                    0.5 * similarity +
                    0.5 * (
                        (2.0 / 8.0) * ocr_confidence +
                        (3.0 / 8.0) * asr_confidence +
                        (3.0 / 8.0) * face_confidence
                    )
                ) AS combined_score
            FROM joined
            WHERE face_match + asr_match + ocr_match > 0
            ORDER BY combined_score DESC
            LIMIT 20;
        """, (
            query_embedding.tolist(),
            emotion.lower(),
            emotion.lower(),
            emotion.lower()
        ))

        result = cursor.fetchall()
        cursor.close()

        response = []
        seen_videos = set()
        for row in result:
            (
                embedding_id,
                location,
                frame_time,
                frame_location,
                similarity,
                annotated_image,
                face_emotion, face_confidence,
                asr_emotion, asr_confidence,
                ocr_emotion, ocr_confidence,
                combined_score
            ) = row

            full_path = os.path.join(dir_1, location)

            if not os.path.exists(full_path):
                continue

            if not allow_duplicates:
                if full_path in seen_videos:
                    continue
                seen_videos.add(full_path)

            response.append({
                "embedding_id": embedding_id,
                "video_path": full_path,
                "frame_time": float(frame_time),
                "similarity": round(float(similarity), 3),
                "final_score": round(float(combined_score), 3),
                "annotated_image": annotated_image if annotated_image else None,
                "frame_location": frame_location,
                "face_emotion": face_emotion,
                "face_confidence": round(float(face_confidence), 3),
                "asr_emotion": asr_emotion,
                "asr_confidence": round(float(asr_confidence), 3),
                "ocr_emotion": ocr_emotion,
                "ocr_confidence": round(float(ocr_confidence), 3)
            })

        return JSONResponse(response)

    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)



@app.get("/")
async def read_root(request: Request):
    """ For testing purposes there exists an index.html file. Not used in app. """
    return templates.TemplateResponse("index.html", context={"request": request})

@app.get("/video")
async def video_endpoint(path: str, start_time: float = 0.0, range: str = Header(None)):
    """
    Allows video streaming over http. Starts at given time.
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


def emotion_mapping(emotion:str):
    """ EMotion mapping, because models have diff. outputs. """
    emotion_to_emotion = {
        "happy": "joy",
        "sad": "sadness",
        "surprise": "surprise",
        "angry": "anger",
        "neutral": "neutral",
        "disgust": "disgust",
        "fear": "fear"
    }
    return emotion_to_emotion.get(emotion.lower(), "neutral")


@app.get("/ask_llama/{query}/{emotion}/{allow_duplicates}")
async def send_query_to_llama(query: str, emotion:str, allow_duplicates: bool):
    """ Query Expansion through llama. New search results are also queried in this method. """
    response = await run_in_threadpool(
        requests.post,
        "http://localhost:11434/api/generate",
        json={
            "model": "tinyllama",
            "prompt": f"Give only 5 to 7 short and relevant keywords (synonyms and similar words) related to this query:\n\n{query}\n\nRespond with a comma-separated list, nothing else.",
            "num_predict": 30,
            "stream": False
        }
    )

    if response.status_code == 200:
        query = response.json().get("response", "").strip()
        cursor = conn.cursor()
        emotion2 = emotion_mapping(emotion)
        print(emotion2)
        try:
            query_embedding = get_embedding(input_text=query.lower())
            query_embedding = normalize_embedding(query_embedding)

            cursor.execute("""
                    WITH top_embeddings AS (
                        SELECT 
                            me.id AS embedding_id,
                            mo.location,
                            me.frame_time,
                            me.frame_location,
                            1 - (me.embedding <=> %s::vector) AS similarity
                        FROM multimedia_embeddings me
                        JOIN multimedia_objects mo ON mo.object_id = me.object_id
                        ORDER BY similarity DESC
                        LIMIT 800
                    ),
                    joined AS (
                        SELECT 
                            te.embedding_id,
                            te.location,
                            te.frame_time,
                            te.frame_location,
                            te.similarity,
                            COALESCE(f.emotion, '') AS face_emotion,
                            COALESCE(f.confidence, 0.0) AS face_confidence,
                            COALESCE(a.emotion, '') AS asr_emotion,
                            COALESCE(a.confidence, 0.0) AS asr_confidence,
                            COALESCE(o.emotion, '') AS ocr_emotion,
                            COALESCE(o.sentiment_confidence, 0.0) AS ocr_confidence,
                            COALESCE(f.path_annotated_faces, '') AS annotated_image,
                            CASE WHEN LOWER(f.emotion) = LOWER(%s) THEN 1 ELSE 0 END AS face_match,
                            CASE WHEN LOWER(a.emotion) = LOWER(%s) THEN 1 ELSE 0 END AS asr_match,
                            CASE WHEN LOWER(o.emotion) = LOWER(%s) THEN 1 ELSE 0 END AS ocr_match
                        FROM top_embeddings te
                        LEFT JOIN Face f ON f.embedding_id = te.embedding_id
                        LEFT JOIN ASR a ON a.embedding_id = te.embedding_id
                        LEFT JOIN OCR o ON o.embedding_id = te.embedding_id
                    )
                    SELECT 
                        embedding_id,
                        location,
                        frame_time,
                        frame_location,
                        similarity,
                        annotated_image,
                        face_emotion, face_confidence,
                        asr_emotion, asr_confidence,
                        ocr_emotion, ocr_confidence,
                        (
                            0.5 * similarity +
                            0.5 * (
                                (2.0 / 8.0) * ocr_confidence +
                                (3.0 / 8.0) * asr_confidence +
                                (3.0 / 8.0) * face_confidence
                            )
                        ) AS combined_score
                    FROM joined
                    WHERE face_match + asr_match + ocr_match > 0
                    ORDER BY combined_score DESC
                    LIMIT 20;
                """, (
                query_embedding.tolist(),
                emotion.lower(),
                emotion.lower(),
                emotion.lower()
            ))

            result = cursor.fetchall()
            cursor.close()

            response = []
            seen_videos = set()
            for row in result:
                (
                    embedding_id,
                    location,
                    frame_time,
                    frame_location,
                    similarity,
                    annotated_image,
                    face_emotion, face_confidence,
                    asr_emotion, asr_confidence,
                    ocr_emotion, ocr_confidence,
                    combined_score
                ) = row

                full_path = os.path.join(dir_1, location)

                if not os.path.exists(full_path):
                    continue

                if not allow_duplicates:
                    if full_path in seen_videos:
                        continue
                    seen_videos.add(full_path)

                response.append({
                    "llama_updated_query": query,
                    "embedding_id": embedding_id,
                    "video_path": full_path,
                    "frame_time": float(frame_time),
                    "similarity": round(float(similarity), 3),
                    "final_score": round(float(combined_score), 3),
                    "annotated_image": annotated_image if annotated_image else None,
                    "frame_location": frame_location,
                    "face_emotion": face_emotion,
                    "face_confidence": round(float(face_confidence), 3),
                    "asr_emotion": asr_emotion,
                    "asr_confidence": round(float(asr_confidence), 3),
                    "ocr_emotion": ocr_emotion,
                    "ocr_confidence": round(float(ocr_confidence), 3)
                })

            return JSONResponse(response)

        except Exception as e:
            print(traceback.format_exc())
            return JSONResponse({"error": str(e)}, status_code=500)
    else:
        raise Exception(f"Failed to get response from LLaMA. Status code: {response.status_code}")



@app.get("/search_by_direction_pair/{datatype}/{emotion}/{allow_duplicates}")
async def search_by_direction_pair(source_id: int, target_id: int, datatype: str, emotion: str, allow_duplicates: bool):
    """
    Computes direction vector from source to target embedding,
    filters results by datatype and emotion that are given by the user in the app.
    """
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT embedding FROM multimedia_embeddings WHERE id = %s", (source_id,))
        row_a = cursor.fetchone()

        cursor.execute("SELECT embedding FROM multimedia_embeddings WHERE id = %s", (target_id,))
        row_b = cursor.fetchone()

        if not row_a or not row_b:
            raise HTTPException(status_code=404, detail="Embeddings not found.")

        emb_a = np.array(row_a[0])
        emb_b = np.array(row_b[0])
        direction = normalize_embedding(emb_b - emb_a)
        projected = normalize_embedding(emb_b + direction)

        if datatype == "all":
            filter_query = """
                SELECT 
                    me.id,
                    (SELECT location FROM multimedia_objects WHERE object_id = me.object_id) AS location,
                    me.frame_time,
                    frame_location, 
                    1 - (me.embedding <=> %s::vector) AS similarity,
                    COALESCE(
                        (SELECT path_annotated_location FROM OCR WHERE embedding_id = me.id LIMIT 1),
                        (SELECT path_annotated_faces FROM Face WHERE embedding_id = me.id LIMIT 1)
                    ) AS annotated_image
                FROM multimedia_embeddings me
                ORDER BY similarity DESC
                LIMIT 20;
            """
            cursor.execute(filter_query, (projected.tolist(),))
            results = cursor.fetchall()

        else:
            if datatype == "face":
                join_table = "Face f"
                emotion_col = "f.emotion"
                image_col = "f.path_annotated_faces"
                join_condition = "f.embedding_id = me.id"
            elif datatype == "ocr":
                join_table = "OCR o"
                emotion_col = "o.emotion"
                image_col = "o.path_annotated_location"
                join_condition = "o.embedding_id = me.id"
            elif datatype == "asr":
                join_table = "ASR a"
                emotion_col = "a.emotion"
                image_col = "NULL"
                join_condition = "a.embedding_id = me.id"
            else:
                raise HTTPException(status_code=400, detail="Invalid datatype")

            query = f"""
                SELECT 
                    me.id,
                    (SELECT location FROM multimedia_objects WHERE object_id = me.object_id) AS location,
                    me.frame_time,
                    me.frame_location,
                    1 - (me.embedding <=> %s::vector) AS similarity,
                    {image_col} AS annotated_image
                FROM multimedia_embeddings me
                JOIN {join_table} ON {join_condition}
                WHERE ({emotion_col} IS NOT NULL AND LOWER({emotion_col}) = LOWER(%s))
                ORDER BY similarity DESC
                LIMIT 20;
            """
            cursor.execute(query, (projected.tolist(), emotion))
            results = cursor.fetchall()

            if len(results) == 0:
                cursor.execute("""
                    SELECT 
                        me.id,
                        (SELECT location FROM multimedia_objects WHERE object_id = me.object_id) AS location,
                        me.frame_time,
                        me.frame_location,
                        1 - (me.embedding <=> %s::vector) AS similarity,
                        COALESCE(
                            (SELECT path_annotated_location FROM OCR WHERE embedding_id = me.id LIMIT 1),
                            (SELECT path_annotated_faces FROM Face WHERE embedding_id = me.id LIMIT 1)
                        ) AS annotated_image
                    FROM multimedia_embeddings me
                    ORDER BY similarity DESC
                    LIMIT 20;
                """, (projected.tolist(),))
                results = cursor.fetchall()

        cursor.close()

        response = []
        seen_videos = set()

        for row in results:
            embedding_id, location, frame_time, frame_location, similarity, annotated_image = row
            full_path = os.path.join(VIDEO_DIRECTORY, location)

            if not os.path.exists(full_path):
                continue

            if not allow_duplicates:
                if full_path in seen_videos:
                    continue
                seen_videos.add(full_path)

            response.append({
                "embedding_id": embedding_id,
                "video_path": full_path,
                "frame_time": frame_time,
                "similarity": float(similarity),
                "annotated_image": str(annotated_image) if annotated_image else None,
                "frame_location": frame_location
            })

        return JSONResponse(response)

    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)