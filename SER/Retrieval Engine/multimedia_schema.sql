CREATE TABLE multimedia_objects (
object_id SERIAL PRIMARY KEY,
type VARCHAR(30),
location VARCHAR(255)
);

CREATE TABLE multimedia_embeddings (
id SERIAL PRIMARY KEY,
object_id INTEGER,
frame_time FLOAT,
embedding vector(512),

emotion_voice VARCHAR(255),
emotion_percentage_voice FLOAT,
sentiment_voice VARCHAR(255),

emotion_face VARCHAR(255),
emotion_percentage_face FLOAT,
sentiment_face VARCHAR(255),

emotion_ocr VARCHAR(255),
emotion_percentage_ocr FLOAT,
sentiment_ocr VARCHAR(255),
FOREIGN KEY(object_id) REFERENCES multimedia_objects(object_id)
);

CREATE TABLE EMOTIONS (
id SERIAL PRIMARY KEY,
date_time TIMESTAMP,
emotion VARCHAR(255)
)