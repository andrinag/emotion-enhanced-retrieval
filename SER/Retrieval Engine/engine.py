import os
from http.client import HTTPException
from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
import psycopg2
import open_clip
import io
from pgvector.psycopg2 import register_vector
from transformers import CLIPProcessor, CLIPModel
import uvicorn
from fastapi.responses import FileResponse


"""
Extremely simple engine that lets you upload images (and vectors) into a postgres database
"""

folder_path = "images/archive/images"  # the downloaded pokemon dataset
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") # pretrained CLIP model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

app = FastAPI()

# connection to the database
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


@app.post("/upload/")
async def upload_files(files: list[UploadFile] = File(...)):
    """
    upload the embedded images into the database
    :param files: image files as a list
    :return: returns status code
    """
    try:
        cursor = conn.cursor()

        for file in files:
            file_content = file.file.read()
            file.file.seek(0)
            embedding = get_embedding(input_image=file_content)
            cursor.execute(
                "INSERT INTO multimedia (filename, embedding) VALUES (%s, %s) ON CONFLICT (filename) DO NOTHING;",
                (file.filename, embedding.tolist())
            )

        conn.commit()
        cursor.close()
        return {"message": f"Uploaded {len(files)} files successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/")
async def search_images(query: str):
    """
    allows you to search for images
    :param query:
    :return: returns the images
    """
    try:
        cursor = conn.cursor()
        query_embedding = get_embedding(input_text=query)
        cursor.execute("""
            SELECT filename, embedding <#> %s::vector AS distance
            FROM multimedia
            ORDER BY distance ASC
            LIMIT 3;
        """, (query_embedding.tolist(),))

        results = cursor.fetchall()
        cursor.close()
        return {"results": [{"filename": row[0], "distance": row[1]} for row in results]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



IMAGE_FOLDER = "./images/archive/images/"


@app.get("/images/{filename}")
async def get_image(filename: str):
    """
    Retrieves the image name from the DB and then serves the local file in the browser.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM multimedia WHERE filename = %s", (filename,))
    result = cursor.fetchone()
    cursor.close()

    if result:
        image_path = os.path.join(IMAGE_FOLDER, filename)

        if os.path.exists(image_path):
            return FileResponse(image_path, media_type="image/png")
        return {"error": f"Image file {filename} not found on disk"}

    return {"error": f"Image {filename} not found in database"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)